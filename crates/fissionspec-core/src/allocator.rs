use core::fmt;
use core::sync::atomic::{AtomicU64, Ordering};

const FREE_LIST_END: u32 = u32::MAX;
static NEXT_POOL_ID: AtomicU64 = AtomicU64::new(1);

/// Pool-scoped, generation-checked identity for one page slot.
///
/// A handle is valid only in the allocator that issued its `pool_id`. Generations
/// never wrap: a slot that exhausts the `u64` generation space is retired.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PageHandle {
    pool_id: u64,
    index: u32,
    generation: u64,
}

impl PageHandle {
    #[must_use]
    pub const fn pool_id(self) -> u64 {
        self.pool_id
    }

    #[must_use]
    pub const fn index(self) -> u32 {
        self.index
    }

    #[must_use]
    pub const fn generation(self) -> u64 {
        self.generation
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Slot {
    generation: u64,
    next_free: u32,
    allocated: bool,
    retired: bool,
}

impl Slot {
    const EMPTY: Self = Self {
        generation: 1,
        next_free: FREE_LIST_END,
        allocated: false,
        retired: false,
    };
}

/// Failure to release a page handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationError {
    WrongPool {
        expected_pool_id: u64,
        actual_pool_id: u64,
    },
    OutOfRange {
        index: u32,
    },
    StaleHandle {
        index: u32,
        expected_generation: u64,
        actual_generation: u64,
    },
    NotAllocated {
        index: u32,
    },
}

impl fmt::Display for AllocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongPool {
                expected_pool_id,
                actual_pool_id,
            } => write!(
                f,
                "page belongs to pool {actual_pool_id}, not pool {expected_pool_id}"
            ),
            Self::OutOfRange { index } => write!(f, "page index {index} is out of range"),
            Self::StaleHandle {
                index,
                expected_generation,
                actual_generation,
            } => write!(
                f,
                "stale page {index}: handle generation {actual_generation}, current generation {expected_generation}"
            ),
            Self::NotAllocated { index } => write!(f, "page {index} is not allocated"),
        }
    }
}

impl std::error::Error for AllocationError {}

/// O(1), fixed-capacity page metadata allocator.
///
/// Storage is embedded in the value, so allocation and release never allocate
/// memory or take locks. Checked generation advancement on release rejects stale
/// KV-page handles after speculative branch rollback.
#[derive(Debug)]
pub struct PageAllocator<const N: usize> {
    slots: [Slot; N],
    pool_id: u64,
    free_head: u32,
    allocated: usize,
    retired: usize,
}

impl<const N: usize> PageAllocator<N> {
    /// Initializes a free-list over all `N` slots.
    ///
    /// # Panics
    ///
    /// Panics for an unrepresentable capacity greater than `u32::MAX`, or if the
    /// process exhausts its checked `u64` allocator identity space.
    #[must_use]
    pub fn new() -> Self {
        assert!(
            N <= u32::MAX as usize,
            "page capacity exceeds u32 index space"
        );
        let mut slots = [Slot::EMPTY; N];
        for (index, slot) in slots.iter_mut().enumerate() {
            slot.next_free = if index + 1 < N {
                (index + 1) as u32
            } else {
                FREE_LIST_END
            };
        }
        Self {
            slots,
            pool_id: next_pool_id(),
            free_head: if N == 0 { FREE_LIST_END } else { 0 },
            allocated: 0,
            retired: 0,
        }
    }

    /// Pops one page from the free-list.
    #[must_use]
    pub fn allocate(&mut self) -> Option<PageHandle> {
        if self.free_head == FREE_LIST_END {
            return None;
        }
        let index = self.free_head as usize;
        let slot = &mut self.slots[index];
        debug_assert!(!slot.allocated && !slot.retired);
        self.free_head = slot.next_free;
        slot.next_free = FREE_LIST_END;
        slot.allocated = true;
        self.allocated += 1;
        Some(PageHandle {
            pool_id: self.pool_id,
            index: index as u32,
            generation: slot.generation,
        })
    }

    /// Returns a page to the free-list and invalidates all copies of its handle.
    ///
    /// On generation exhaustion the page is safely released but permanently
    /// retired instead of allowing its generation to wrap.
    pub fn release(&mut self, handle: PageHandle) -> Result<(), AllocationError> {
        if handle.pool_id != self.pool_id {
            return Err(AllocationError::WrongPool {
                expected_pool_id: self.pool_id,
                actual_pool_id: handle.pool_id,
            });
        }
        let index = handle.index as usize;
        let Some(slot) = self.slots.get_mut(index) else {
            return Err(AllocationError::OutOfRange {
                index: handle.index,
            });
        };
        if slot.generation != handle.generation {
            return Err(AllocationError::StaleHandle {
                index: handle.index,
                expected_generation: slot.generation,
                actual_generation: handle.generation,
            });
        }
        if !slot.allocated {
            return Err(AllocationError::NotAllocated {
                index: handle.index,
            });
        }

        slot.allocated = false;
        self.allocated -= 1;
        if let Some(generation) = slot.generation.checked_add(1) {
            slot.generation = generation;
            slot.next_free = self.free_head;
            self.free_head = handle.index;
        } else {
            slot.retired = true;
            slot.next_free = FREE_LIST_END;
            self.retired += 1;
        }
        Ok(())
    }

    /// True only if the exact slot generation is currently live.
    #[must_use]
    pub fn contains(&self, handle: PageHandle) -> bool {
        handle.pool_id == self.pool_id
            && self
                .slots
                .get(handle.index as usize)
                .is_some_and(|slot| slot.allocated && slot.generation == handle.generation)
    }

    #[must_use]
    pub const fn pool_id(&self) -> u64 {
        self.pool_id
    }

    #[must_use]
    pub const fn capacity(&self) -> usize {
        N
    }

    #[must_use]
    pub const fn allocated(&self) -> usize {
        self.allocated
    }

    #[must_use]
    pub const fn retired(&self) -> usize {
        self.retired
    }

    #[must_use]
    pub const fn available(&self) -> usize {
        N - self.allocated - self.retired
    }
}

impl<const N: usize> Default for PageAllocator<N> {
    fn default() -> Self {
        Self::new()
    }
}

fn next_pool_id() -> u64 {
    NEXT_POOL_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .expect("page allocator pool identity space exhausted")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_to_capacity_and_reuses_lifo() {
        let mut allocator = PageAllocator::<2>::new();
        let first = allocator.allocate().unwrap();
        let second = allocator.allocate().unwrap();
        assert_eq!((first.index(), second.index()), (0, 1));
        assert_eq!(allocator.allocate(), None);
        assert_eq!(allocator.allocated(), 2);

        allocator.release(first).unwrap();
        let replacement = allocator.allocate().unwrap();
        assert_eq!(replacement.index(), first.index());
        assert_ne!(replacement.generation(), first.generation());
        assert_eq!(allocator.available(), 0);
    }

    #[test]
    fn stale_and_double_release_are_rejected() {
        let mut allocator = PageAllocator::<1>::new();
        let old = allocator.allocate().unwrap();
        allocator.release(old).unwrap();
        assert!(matches!(
            allocator.release(old),
            Err(AllocationError::StaleHandle { .. })
        ));
        let current = allocator.allocate().unwrap();
        assert!(!allocator.contains(old));
        assert!(allocator.contains(current));
    }

    #[test]
    fn handles_are_scoped_to_the_issuing_pool() {
        let mut first_pool = PageAllocator::<1>::new();
        let mut second_pool = PageAllocator::<1>::new();
        let first = first_pool.allocate().unwrap();
        let second = second_pool.allocate().unwrap();

        assert_ne!(first.pool_id(), second.pool_id());
        assert!(matches!(
            second_pool.release(first),
            Err(AllocationError::WrongPool { .. })
        ));
        assert!(second_pool.contains(second));
    }

    #[test]
    fn generation_exhaustion_retires_instead_of_wrapping() {
        let mut allocator = PageAllocator::<1>::new();
        allocator.slots[0].generation = u64::MAX;
        let handle = allocator.allocate().unwrap();
        allocator.release(handle).unwrap();
        assert!(!allocator.contains(handle));
        assert_eq!(allocator.allocate(), None);
        assert_eq!(allocator.allocated(), 0);
        assert_eq!(allocator.available(), 0);
        assert_eq!(allocator.retired(), 1);
    }

    #[test]
    fn zero_capacity_is_valid() {
        let mut allocator = PageAllocator::<0>::new();
        assert_eq!(allocator.capacity(), 0);
        assert_eq!(allocator.allocate(), None);
        assert_eq!(allocator.retired(), 0);
    }
}

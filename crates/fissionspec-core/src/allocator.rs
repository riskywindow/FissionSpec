use core::fmt;

const FREE_LIST_END: u32 = u32::MAX;

/// ABA-resistant identity for one page slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PageHandle {
    index: u32,
    generation: u32,
}

impl PageHandle {
    #[must_use]
    pub const fn index(self) -> u32 {
        self.index
    }

    #[must_use]
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Slot {
    generation: u32,
    next_free: u32,
    allocated: bool,
}

impl Slot {
    const EMPTY: Self = Self {
        generation: 1,
        next_free: FREE_LIST_END,
        allocated: false,
    };
}

/// Failure to release a page handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationError {
    OutOfRange {
        index: u32,
    },
    StaleHandle {
        index: u32,
        expected_generation: u32,
        actual_generation: u32,
    },
    NotAllocated {
        index: u32,
    },
}

impl fmt::Display for AllocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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
/// memory or take locks. A generation increment on release rejects stale KV-page
/// handles after speculative branch rollback.
#[derive(Debug)]
pub struct PageAllocator<const N: usize> {
    slots: [Slot; N],
    free_head: u32,
    allocated: usize,
}

impl<const N: usize> PageAllocator<N> {
    /// Initializes a free-list over all `N` slots.
    ///
    /// # Panics
    ///
    /// Panics only for an unrepresentable capacity greater than `u32::MAX`.
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
            free_head: if N == 0 { FREE_LIST_END } else { 0 },
            allocated: 0,
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
        self.free_head = slot.next_free;
        slot.next_free = FREE_LIST_END;
        slot.allocated = true;
        self.allocated += 1;
        Some(PageHandle {
            index: index as u32,
            generation: slot.generation,
        })
    }

    /// Returns a page to the free-list and invalidates all copies of its handle.
    pub fn release(&mut self, handle: PageHandle) -> Result<(), AllocationError> {
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
        slot.generation = next_generation(slot.generation);
        slot.next_free = self.free_head;
        self.free_head = handle.index;
        self.allocated -= 1;
        Ok(())
    }

    /// True only if the exact slot generation is currently live.
    #[must_use]
    pub fn contains(&self, handle: PageHandle) -> bool {
        self.slots
            .get(handle.index as usize)
            .is_some_and(|slot| slot.allocated && slot.generation == handle.generation)
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
    pub const fn available(&self) -> usize {
        N - self.allocated
    }
}

impl<const N: usize> Default for PageAllocator<N> {
    fn default() -> Self {
        Self::new()
    }
}

const fn next_generation(generation: u32) -> u32 {
    if generation == u32::MAX {
        1
    } else {
        generation + 1
    }
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
    fn generation_wrap_skips_reserved_zero() {
        let mut allocator = PageAllocator::<1>::new();
        allocator.slots[0].generation = u32::MAX;
        let handle = allocator.allocate().unwrap();
        allocator.release(handle).unwrap();
        assert_eq!(allocator.slots[0].generation, 1);
    }

    #[test]
    fn zero_capacity_is_valid() {
        let mut allocator = PageAllocator::<0>::new();
        assert_eq!(allocator.capacity(), 0);
        assert_eq!(allocator.allocate(), None);
    }
}

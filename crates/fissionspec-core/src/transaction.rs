use crate::PageHandle;
use core::fmt;

/// Monotonically increasing scheduler/KV epoch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Epoch(pub u64);

/// Branch identity within an epoch. Branch zero is conventionally the trunk.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BranchId(pub u32);

/// Compact identity carried by recovery work.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TransactionStamp {
    pub epoch: Epoch,
    pub branch: BranchId,
}

impl TransactionStamp {
    #[must_use]
    pub const fn new(epoch: Epoch, branch: BranchId) -> Self {
        Self { epoch, branch }
    }
}

/// State of a speculative branch's page transaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransactionState {
    Open,
    Prepared,
    Committed,
    Aborted,
}

/// Minimal metadata needed to fence and resolve a branch transaction.
///
/// The checkpoint is the last page shared with the parent; the optional tail is
/// the last branch-private page prepared for refusion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransactionMeta {
    stamp: TransactionStamp,
    parent: Option<BranchId>,
    checkpoint: Option<PageHandle>,
    tail: Option<PageHandle>,
    state: TransactionState,
}

impl TransactionMeta {
    #[must_use]
    pub const fn begin(
        stamp: TransactionStamp,
        parent: Option<BranchId>,
        checkpoint: Option<PageHandle>,
    ) -> Self {
        Self {
            stamp,
            parent,
            checkpoint,
            tail: checkpoint,
            state: TransactionState::Open,
        }
    }

    #[must_use]
    pub const fn stamp(&self) -> TransactionStamp {
        self.stamp
    }

    #[must_use]
    pub const fn parent(&self) -> Option<BranchId> {
        self.parent
    }

    #[must_use]
    pub const fn checkpoint(&self) -> Option<PageHandle> {
        self.checkpoint
    }

    #[must_use]
    pub const fn tail(&self) -> Option<PageHandle> {
        self.tail
    }

    #[must_use]
    pub const fn state(&self) -> TransactionState {
        self.state
    }

    /// Freezes a branch tail before an atomic parent-pointer swap.
    pub fn prepare(
        &mut self,
        observed_epoch: Epoch,
        tail: Option<PageHandle>,
    ) -> Result<(), TransactionError> {
        self.check_epoch(observed_epoch)?;
        if self.state != TransactionState::Open {
            return Err(TransactionError::InvalidTransition {
                from: self.state,
                operation: "prepare",
            });
        }
        self.tail = tail;
        self.state = TransactionState::Prepared;
        Ok(())
    }

    /// Marks the branch visible in its declared parent after epoch/parent fencing.
    pub fn commit_into(
        &mut self,
        observed_epoch: Epoch,
        target_parent: Option<BranchId>,
    ) -> Result<(), TransactionError> {
        self.check_epoch(observed_epoch)?;
        if target_parent != self.parent {
            return Err(TransactionError::ParentMismatch {
                expected: self.parent,
                actual: target_parent,
            });
        }
        if !matches!(
            self.state,
            TransactionState::Open | TransactionState::Prepared
        ) {
            return Err(TransactionError::InvalidTransition {
                from: self.state,
                operation: "commit",
            });
        }
        self.state = TransactionState::Committed;
        Ok(())
    }

    /// Rolls back an open or prepared branch. Page reclamation remains the
    /// allocator owner's responsibility.
    pub fn abort(&mut self, observed_epoch: Epoch) -> Result<(), TransactionError> {
        self.check_epoch(observed_epoch)?;
        if !matches!(
            self.state,
            TransactionState::Open | TransactionState::Prepared
        ) {
            return Err(TransactionError::InvalidTransition {
                from: self.state,
                operation: "abort",
            });
        }
        self.state = TransactionState::Aborted;
        Ok(())
    }

    fn check_epoch(&self, observed: Epoch) -> Result<(), TransactionError> {
        if observed == self.stamp.epoch {
            Ok(())
        } else {
            Err(TransactionError::EpochMismatch {
                expected: self.stamp.epoch,
                actual: observed,
            })
        }
    }
}

/// Invalid transaction operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransactionError {
    EpochMismatch {
        expected: Epoch,
        actual: Epoch,
    },
    ParentMismatch {
        expected: Option<BranchId>,
        actual: Option<BranchId>,
    },
    InvalidTransition {
        from: TransactionState,
        operation: &'static str,
    },
}

impl fmt::Display for TransactionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EpochMismatch { expected, actual } => write!(
                f,
                "epoch fence failed: expected {}, observed {}",
                expected.0, actual.0
            ),
            Self::ParentMismatch { expected, actual } => {
                write!(
                    f,
                    "parent fence failed: expected {expected:?}, observed {actual:?}"
                )
            }
            Self::InvalidTransition { from, operation } => {
                write!(f, "cannot {operation} a {from:?} transaction")
            }
        }
    }
}

impl std::error::Error for TransactionError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PageAllocator;

    #[test]
    fn prepared_branch_commits_into_declared_parent() {
        let mut pages = PageAllocator::<2>::new();
        let checkpoint = pages.allocate();
        let tail = pages.allocate();
        let stamp = TransactionStamp::new(Epoch(7), BranchId(3));
        let mut transaction = TransactionMeta::begin(stamp, Some(BranchId(1)), checkpoint);

        transaction.prepare(Epoch(7), tail).unwrap();
        assert_eq!(transaction.state(), TransactionState::Prepared);
        assert_eq!(transaction.tail(), tail);
        transaction
            .commit_into(Epoch(7), Some(BranchId(1)))
            .unwrap();
        assert_eq!(transaction.state(), TransactionState::Committed);
    }

    #[test]
    fn epoch_and_parent_fences_reject_stale_refusion() {
        let stamp = TransactionStamp::new(Epoch(9), BranchId(2));
        let mut transaction = TransactionMeta::begin(stamp, Some(BranchId(1)), None);
        assert!(matches!(
            transaction.prepare(Epoch(8), None),
            Err(TransactionError::EpochMismatch { .. })
        ));
        assert!(matches!(
            transaction.commit_into(Epoch(9), Some(BranchId(4))),
            Err(TransactionError::ParentMismatch { .. })
        ));
        assert_eq!(transaction.state(), TransactionState::Open);
    }

    #[test]
    fn terminal_states_cannot_transition_again() {
        let stamp = TransactionStamp::new(Epoch(1), BranchId(2));
        let mut transaction = TransactionMeta::begin(stamp, None, None);
        transaction.abort(Epoch(1)).unwrap();
        assert!(matches!(
            transaction.commit_into(Epoch(1), None),
            Err(TransactionError::InvalidTransition { .. })
        ));
        assert!(matches!(
            transaction.abort(Epoch(1)),
            Err(TransactionError::InvalidTransition { .. })
        ));
    }
}

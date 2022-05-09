#![no_std]

use core::{
    fmt,
    iter::{ExactSizeIterator, FromIterator, FusedIterator},
    mem,
    ops::{Bound, Not, RangeBounds},
};

use num_traits::{Bounded, PrimInt};

/// A bit set able to hold up to 8 elements
pub type BitSet8 = BitSet<u8, 1>;
/// A bit set able to hold up to 16 elements
pub type BitSet16 = BitSet<u16, 1>;
/// A bit set able to hold up to 32 elements
pub type BitSet32 = BitSet<u32, 1>;
/// A bit set able to hold up to 64 elements
pub type BitSet64 = BitSet<u64, 1>;
/// A bit set able to hold up to 128 elements
pub type BitSet128 = BitSet<u64, 2>;
/// A bit set able to hold up to 256 elements
pub type BitSet256 = BitSet<u64, 4>;
/// A bit set able to hold up to 512 elements
pub type BitSet512 = BitSet<u64, 8>;
/// A bit set able to hold up to 1024 elements
pub type BitSet1024 = BitSet<u64, 16>;

/// The bit set itself
///
/// This wrapper is `#![repr(transparent)]` and guaranteed to have the same memory
/// representation as the inner bit array
///
/// # Panics
/// All non-try functions taking a bit parameter panics if the bit is bigger
/// than the capacity of the set. For non-panicking versions, use `try_`.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BitSet<T, const N: usize> {
    inner: [T; N],
}

impl<T: PrimInt + Default, const N: usize> Default for BitSet<T, N> {
    fn default() -> Self {
        Self {
            inner: [Default::default(); N],
        }
    }
}

impl<T: PrimInt, const N: usize> From<[T; N]> for BitSet<T, N> {
    fn from(inner: [T; N]) -> Self {
        Self { inner }
    }
}

impl<T, const N: usize> fmt::Debug for BitSet<T, N>
where T: Copy + Clone + fmt::Binary
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "BitSet ")?;
        let mut list = f.debug_list();

        for item in self.inner.iter() {
            list.entry(&format_args!(
                "{:#0width$b}",
                item,
                width = 2 /* 0b */ + Self::item_size()
            ));
        }

        list.finish()
    }
}

/// Removed hacking macros just to add another one LOL
/// FIXME: Use const trait implementation when that is stabilized
macro_rules! impl_new {
    ($($t:ty)+) => {
        $(
        impl<const N: usize> BitSet<$t, N> {
            /// Create an empty instance of [`BitSet`]
            pub const fn new() -> Self {
                Self { inner: [0; N] }
            }
        }
        )+
    };
}

impl_new!(i8 i16 i32 i64 i128 isize);
impl_new!(u8 u16 u32 u64 u128 usize);

impl<T: PrimInt + Default, const N: usize> BitSet<T, N> {
    /// Create an empty instance with default value
    ///
    /// This function is the same as [`new`](BitSet::new) but without the constness.
    pub fn with_default() -> Self {
        Self::default()
    }

    /// Disable all bits, probably faster than what `fill(.., false)` would do
    pub fn clear(&mut self) {
        for item in self.inner.iter_mut() {
            *item = Default::default()
        }
    }
}

impl<T, const N: usize> BitSet<T, N> {
    /// Return the inner integer array
    pub fn into_inner(self) -> [T; N] {
        self.inner
    }

    /// Returns the capacity of the set, in other words how many bits it can
    /// hold. This function may very well overflow if the size or length is too
    /// big, but if you're making that big allocations you probably got bigger
    /// things to worry about.
    pub const fn capacity() -> usize {
        N * Self::item_size()
    }

    /// Returns the bit size of each item
    const fn item_size() -> usize {
        mem::size_of::<T>() * 8
    }
}

impl<T: PrimInt, const N: usize> BitSet<T, N> {
    /// Transmutes a reference to a borrowed bit array to a borrowed BitSet
    /// with the same lifetime
    pub fn from_ref(inner: &mut [T; N]) -> &mut Self {
        // This should be completely safe as the memory representation is the
        // same
        unsafe { mem::transmute(inner) }
    }

    /// Returns slot index along with the bitmask for the bit
    /// index to the slot this item was in
    fn location(bit: usize) -> (usize, T) {
        let index = bit / Self::item_size();
        let bitmask = T::one() << (bit & (Self::item_size() - 1));
        (index, bitmask)
    }

    /// Like `insert`, but does not panic if the bit is too large. See
    /// the struct level documentation for notes on panicking.
    pub fn try_insert(&mut self, bit: usize) -> bool {
        if bit >= Self::capacity() {
            return false;
        }
        let (index, bitmask) = Self::location(bit);
        match self.inner.get_mut(index) {
            Some(v) => *v = (*v) | bitmask,
            None => (),
        }
        true
    }

    /// Like `remove`, but does not panic if the bit is too large.
    /// See the struct level documentation for notes on panicking.
    pub fn try_remove(&mut self, bit: usize) -> bool {
        if bit >= Self::capacity() {
            return false;
        }
        let (index, bitmask) = Self::location(bit);
        match self.inner.get_mut(index) {
            Some(v) => *v = (*v) & !bitmask,
            None => (),
        }
        true
    }

    /// Enable the specified bit in the set. If the bit is already
    /// enabled this is a no-op.
    pub fn insert(&mut self, bit: usize) {
        assert!(
            self.try_insert(bit),
            "BitSet::insert called on an integer bigger than capacity"
        );
    }

    /// Disable the specified bit in the set. If the bit is already
    /// disabled this is a no-op.
    pub fn remove(&mut self, bit: usize) {
        assert!(
            self.try_remove(bit),
            "BitSet::remove called on an integer bigger than capacity"
        );
    }

    /// Returns true if the specified bit is enabled. If the bit is
    /// out of bounds this silently returns false.
    pub fn contains(&self, bit: usize) -> bool {
        self.try_contains(bit).unwrap_or(false)
    }

    /// Returns true if the specified bit is enabled
    pub fn try_contains(&self, bit: usize) -> Option<bool> {
        if bit >= Self::capacity() {
            return None;
        }

        let (index, bitmask) = Self::location(bit);
        Some(*self.inner.get(index)? & bitmask == bitmask)
    }

    /// Returns the total number of enabled bits
    pub fn count_ones(&self) -> u32 {
        let mut total = 0;
        for item in self.inner.iter() {
            total += item.count_ones();
        }
        total
    }

    /// Returns the total number of disabled bits
    pub fn count_zeros(&self) -> u32 {
        let mut total = 0;

        for item in self.inner.iter() {
            total += item.count_zeros();
        }

        total
    }
}

impl<T: Default + PrimInt, const N: usize> BitSet<T, N> {
    /// Set all bits in a range.
    /// `fill(.., false)` is effectively the same as `clear()`.
    ///
    /// # Panics
    /// Panics if the start or end bounds are more than the capacity.
    pub fn fill<R: RangeBounds<usize>>(&mut self, range: R, on: bool) {
        let mut start = match range.start_bound() {
            Bound::Unbounded => 0,
            Bound::Included(&i) => {
                assert!(i <= Self::capacity(), "start bound is too big for capacity");
                i
            },
            Bound::Excluded(&i) => {
                assert!(i < Self::capacity(), "start bound is too big for capacity");
                i + 1
            },
        };
        let end = match range.end_bound() {
            Bound::Unbounded => Self::capacity(),
            Bound::Included(0) => return,
            Bound::Included(&i) => {
                assert!(
                    i - 1 <= Self::capacity(),
                    "end bound is too big for capacity"
                );
                i - 1
            },
            Bound::Excluded(&i) => {
                assert!(i <= Self::capacity(), "end bound is too big for capacity");
                i
            },
        };

        if start >= end {
            return;
        }

        let end_first = start - (start % Self::item_size()) + Self::item_size();
        if start % Self::item_size() != 0 || end < end_first {
            // Unaligned write to either the end or the start of next integer
            let end_first = end_first.min(end);
            // println!("Doing initial unaligned from {} to {}", start, end_first);
            for bit in start..end_first {
                if on {
                    self.insert(bit);
                } else {
                    self.remove(bit);
                }
            }

            if end == end_first {
                return;
            }

            start = end_first + 1;
        }

        // Fast way to fill all bits in all integers: Just set them to the min/max value.
        let start_last = end - (end % Self::item_size());
        // println!("Doing aligned from {} to {}", start, start_last);
        for i in start / Self::item_size()..start_last / Self::item_size() {
            self.inner[i] = if on {
                Bounded::max_value()
            } else {
                Default::default()
            };
        }

        // Unaligned write to the end
        // println!("Doing unaligned from {} to {}", start_last, end);
        for bit in start_last..end {
            if on {
                self.insert(bit);
            } else {
                self.remove(bit);
            }
        }
    }
}

impl<T: PrimInt + Default, U: Into<usize>, const N: usize> FromIterator<U> for BitSet<T, N> {
    fn from_iter<I>(iter: I) -> Self
    where I: IntoIterator<Item = U> {
        let mut set = BitSet::with_default();
        for bit in iter.into_iter() {
            set.insert(bit.into());
        }
        set
    }
}

impl<T: PrimInt, U: Into<usize>, const N: usize> Extend<U> for BitSet<T, N> {
    fn extend<I: IntoIterator<Item = U>>(&mut self, iter: I) {
        for bit in iter.into_iter() {
            self.insert(bit.into());
        }
    }
}

impl<T: PrimInt, const N: usize> IntoIterator for BitSet<T, N> {
    type IntoIter = IntoIter<T, N>;
    type Item = usize;

    fn into_iter(self) -> Self::IntoIter {
        crate::IntoIter(self)
    }
}

impl<T: PrimInt, const N: usize> Not for BitSet<T, N> {
    type Output = Self;

    fn not(mut self) -> Self::Output {
        for item in self.inner.iter_mut() {
            *item = !(*item);
        }

        self
    }
}

#[derive(Clone)]
#[repr(transparent)]
pub struct IntoIter<T, const N: usize>(BitSet<T, N>);

impl<T, const N: usize> fmt::Debug for IntoIter<T, N>
where T: Copy + Clone + fmt::Binary
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut list = f.debug_list();

        for item in self.0.inner.iter() {
            list.entry(&format_args!(
                "{:#0width$b}",
                item,
                width = 2 /* 0b */ + BitSet::<T, N>::item_size()
            ));
        }

        list.finish()
    }
}

impl<T: PrimInt, const N: usize> Iterator for IntoIter<T, N> {
    type Item = usize;

    /// Iterator implementation for BitSet, guaranteed to remove and
    /// return the items in ascending order
    fn next(&mut self) -> Option<Self::Item> {
        for (index, item) in self.0.inner.iter_mut().enumerate() {
            if !item.is_zero() {
                let bitindex = item.trailing_zeros() as usize;

                // E.g. 1010 & 1001 = 1000
                *item = *item & (*item - T::one());

                // Safe from overflows because one couldn't possibly add an item with this index if
                // it did overflow
                return Some(index * BitSet::<T, N>::item_size() + bitindex);
            }
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.0.count_ones() as usize;
        (len, Some(len))
    }
}

impl<T: PrimInt, const N: usize> DoubleEndedIterator for IntoIter<T, N> {
    /// Reversed iterator implementation for BitSet, guaranteed to
    /// remove and return the items in descending order
    fn next_back(&mut self) -> Option<Self::Item> {
        for (index, item) in self.0.inner.iter_mut().enumerate().rev() {
            if !item.is_zero() {
                let bitindex = BitSet::<T, N>::item_size() - 1 - item.leading_zeros() as usize;

                // E.g. 00101 & 11011 = 00001, same as remove procedure but using relative index
                *item = *item & !(T::one() << bitindex);

                // Safe from overflows because one couldn't possibly add an item with this index if
                // it did overflow
                return Some(index * BitSet::<T, N>::item_size() + bitindex);
            }
        }
        None
    }
}

impl<T: PrimInt, const N: usize> FusedIterator for IntoIter<T, N> {}
impl<T: PrimInt, const N: usize> ExactSizeIterator for IntoIter<T, N> {}


#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;

    #[test]
    fn repr() {
        assert_eq!(mem::size_of::<BitSet8>(), 1);
        assert_eq!(mem::size_of::<BitSet16>(), 2);
        assert_eq!(mem::size_of::<BitSet32>(), 4);
        assert_eq!(mem::size_of::<BitSet64>(), 8);
        assert_eq!(mem::size_of::<BitSet128>(), 16);
        assert_eq!(mem::size_of::<BitSet256>(), 32);
        assert_eq!(mem::size_of::<BitSet512>(), 64);
    }

    #[test]
    fn capacity() {
        assert_eq!(BitSet8::capacity(), 8);
        assert_eq!(BitSet16::capacity(), 16);
        assert_eq!(BitSet32::capacity(), 32);
        assert_eq!(BitSet64::capacity(), 64);
        assert_eq!(BitSet128::capacity(), 128);
        assert_eq!(BitSet256::capacity(), 256);
        assert_eq!(BitSet512::capacity(), 512);
    }

    #[test]
    fn try_too_big() {
        let mut set = BitSet8::new();
        assert!(!set.try_insert(8));
    }

    #[test]
    #[should_panic]
    fn panic_too_big() {
        let mut set = BitSet128::new();
        set.insert(128);
    }

    #[test]
    fn insert() {
        let mut set = BitSet128::new();
        set.insert(0);
        set.insert(12);
        set.insert(67);
        set.insert(82);
        set.insert(127);
        assert!(set.contains(0));
        assert!(set.contains(12));
        assert!(!set.contains(51));
        assert!(!set.contains(63));
        assert!(set.contains(67));
        assert!(!set.contains(73));
        assert!(set.contains(82));
        assert!(set.contains(127));
    }

    #[test]
    fn remove() {
        let mut set = BitSet32::new();
        set.insert(12);
        set.insert(17);
        assert!(set.contains(12));
        assert!(set.contains(17));
        set.remove(17);
        assert!(set.contains(12));
        assert!(!set.contains(17));
    }

    #[test]
    fn clear() {
        let mut set = BitSet64::new();
        set.insert(35);
        set.insert(42);
        assert!(set.contains(35));
        assert!(set.contains(42));
        set.clear();
        assert!(!set.contains(35));
        assert!(!set.contains(42));
    }

    #[test]
    fn count_ones_and_zeros() {
        let mut set = BitSet8::new();
        set.insert(5);
        set.insert(7);
        assert_eq!(set.count_ones(), 2);
        assert_eq!(set.count_zeros(), 8 - 2);
    }

    #[test]
    fn fill() {
        // Care must be taken when changing the `range` function, as this test
        // won't detect if it actually does as many aligned writes as it can.

        let mut set = BitSet::<u8, 2>::new();

        // Aligned
        set.fill(.., true);
        for i in 0..16 {
            assert!(set.contains(i));
        }

        // println!("---");

        // Within the same int
        set.clear();
        set.fill(1..3, true);
        assert!(!set.contains(0));
        assert!(set.contains(1));
        assert!(set.contains(2));
        assert!(!set.contains(3));

        // println!("---");

        // Unaligned end
        set.clear();
        set.fill(8..10, true);
        assert!(!set.contains(7));
        assert!(set.contains(8));
        assert!(set.contains(9));
        assert!(!set.contains(10));

        // println!("---");

        // Unaligned start
        set.clear();
        set.fill(3..16, true);
        assert!(!set.contains(2));
        for i in 3..16 {
            assert!(set.contains(i));
        }
    }

    #[test]
    fn iter() {
        let set: BitSet<u8, 4> = [30u8, 0, 4, 2, 12, 22, 23, 29].iter().copied().collect();
        let mut iter = set.into_iter();
        assert_eq!(iter.len(), 8);
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.len(), 7);
        assert_eq!(iter.next_back(), Some(30));
        assert_eq!(iter.len(), 6);
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.len(), 5);
        assert_eq!(iter.next_back(), Some(29));
        assert_eq!(iter.len(), 4);
        assert_eq!(iter.next(), Some(4));
        assert_eq!(iter.len(), 3);
        assert_eq!(iter.next_back(), Some(23));
        assert_eq!(iter.len(), 2);
        assert_eq!(iter.next(), Some(12));
        assert_eq!(iter.len(), 1);
        assert_eq!(iter.next_back(), Some(22));
        assert_eq!(iter.len(), 0);
        assert_eq!(iter.next_back(), None);
        assert_eq!(iter.len(), 0);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn debug() {
        use self::alloc::format;
        assert_eq!(
            format!("{:?}", (0u16..10).collect::<BitSet16>()),
            "BitSet [0b0000001111111111]"
        );
        assert_eq!(
            format!("{:#?}", (0u16..10).collect::<BitSet16>()),
            "BitSet [\n    0b0000001111111111,\n]"
        );
    }

    #[test]
    fn not() {
        assert_eq!(
            (0u16..10).collect::<BitSet16>(),
            !(10u16..16).collect::<BitSet16>()
        );
        assert_eq!(
            (10u16..16).collect::<BitSet16>(),
            !(0u16..10).collect::<BitSet16>()
        );
    }
}

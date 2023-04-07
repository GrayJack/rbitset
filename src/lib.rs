#![no_std]
#![cfg_attr(_doc, feature(doc_cfg))]

use core::{
    fmt,
    iter::{Chain, ExactSizeIterator, FromIterator, FusedIterator},
    mem,
    ops::{Bound, Not, RangeBounds},
};

use num_traits::{Bounded, PrimInt};

#[cfg(feature = "serde")]
use serde::{de::Visitor, ser::SerializeSeq, Deserialize, Serialize};

/// A bit set able to hold up to 8 elements.
pub type BitSet8 = BitSet<u8, 1>;
/// A bit set able to hold up to 16 elements.
pub type BitSet16 = BitSet<u16, 1>;
/// A bit set able to hold up to 32 elements.
pub type BitSet32 = BitSet<u32, 1>;
/// A bit set able to hold up to 64 elements.
pub type BitSet64 = BitSet<u64, 1>;
/// A bit set able to hold up to 128 elements.
pub type BitSet128 = BitSet<u64, 2>;
/// A bit set able to hold up to 256 elements.
pub type BitSet256 = BitSet<u64, 4>;
/// A bit set able to hold up to 512 elements.
pub type BitSet512 = BitSet<u64, 8>;
/// A bit set able to hold up to 1024 elements.
pub type BitSet1024 = BitSet<u64, 16>;

/// The bit set itself.
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
where T: PrimInt
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<T, const N: usize> fmt::Binary for BitSet<T, N>
where T: Copy + fmt::Binary
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
            /// Create an empty instance of [`BitSet`].
            ///
            /// # Examples
            ///
            /// ```
            /// use rbitset::BitSet;
            ///
            /// let set = BitSet::<u8, 1>::new();
            /// ```
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
    /// Create an empty instance with default value.
    ///
    /// This function is the same as [`new`](BitSet::new) but without the constness.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbitset::BitSet;
    ///
    /// let set = BitSet::<u32, 7>::new();
    /// ```
    pub fn with_default() -> Self {
        Self::default()
    }

    /// Clears the set, disabling all bits, removing all elements.
    ///
    /// Probably faster than what `fill(.., false)` would be.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbitset::BitSet8;
    ///
    /// let mut set = BitSet8::new();
    /// set.insert(1);
    /// assert!(!set.is_empty());
    /// set.clear();
    /// assert!(set.is_empty());
    /// ```
    pub fn clear(&mut self) {
        for item in self.inner.iter_mut() {
            *item = Default::default()
        }
    }
}

impl<T, const N: usize> BitSet<T, N> {
    /// Return the inner integer array.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbitset::BitSet8;
    ///
    /// let set = BitSet8::from_iter([1u8, 2, 3]);
    /// assert_eq!(set.into_inner(), [0b00001110]);
    /// ```
    pub fn into_inner(self) -> [T; N] {
        self.inner
    }

    /// Returns the capacity of the set, in other words how many bits it can hold.
    ///
    /// This function may very well overflow if the size or length is too big, but if you're making
    /// that big allocations you probably got bigger things to worry about.
    ///
    /// # Examples
    /// ```
    /// use rbitset::BitSet;
    ///
    /// let capacity = BitSet::<u32, 3>::capacity();
    /// assert_eq!(capacity, 32 * 3);
    /// ```
    pub const fn capacity() -> usize {
        N * Self::item_size()
    }

    /// Returns the bit size of each item.
    const fn item_size() -> usize {
        mem::size_of::<T>() * 8
    }
}

impl<T: PrimInt, const N: usize> BitSet<T, N> {
    /// Transmutes a reference to a borrowed bit array to a borrowed BitSet with the same lifetime.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbitset::BitSet;
    ///
    /// let mut raw = [0b00001110, 0u8];
    /// let set = BitSet::from_ref(&mut raw);
    /// assert!(set.contains(1));
    /// assert!(set.contains(2));
    /// assert!(set.contains(3));
    /// ```
    pub fn from_ref(inner: &mut [T; N]) -> &mut Self {
        // This should be completely safe as the memory representation is the same
        unsafe { mem::transmute(inner) }
    }

    /// Returns slot index along with the bitmask for the bit index to the slot this item was in.
    fn location(bit: usize) -> (usize, T) {
        let index = bit / Self::item_size();
        let bitmask = T::one() << (bit & (Self::item_size() - 1));
        (index, bitmask)
    }

    /// Tries to move all elements from `other` into `self`, leaving `other` empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbitset::BitSet16;
    ///
    /// let mut a = BitSet16::new();
    /// a.insert(1);
    /// a.insert(2);
    /// a.insert(3);
    ///
    /// let mut b = BitSet16::new();
    /// b.insert(3);
    /// b.insert(4);
    /// b.insert(5);
    ///
    /// a.try_append(&mut b).expect("An error occurred");
    ///
    /// assert_eq!(a.len(), 5);
    /// assert_eq!(b.len(), 0);
    ///
    /// assert!(a.contains(1));
    /// assert!(a.contains(2));
    /// assert!(a.contains(3));
    /// assert!(a.contains(4));
    /// assert!(a.contains(5));
    /// ```
    pub fn try_append<U, const M: usize>(
        &mut self, other: &mut BitSet<U, M>,
    ) -> Result<(), BitSetError>
    where U: PrimInt {
        for item in other.drain() {
            self.try_insert(item)?;
        }
        Ok(())
    }

    /// Tries to add a value to the set.
    ///
    /// If the set did not have this value present, `true` is returned.
    ///
    /// If the set did have this value present, `false` is returned.
    ///
    /// # Examples
    /// ```
    /// use rbitset::{BitSet16, BitSetError};
    ///
    /// let mut set = BitSet16::new();
    ///
    /// assert_eq!(set.try_insert(2), Ok(true));
    /// assert_eq!(set.try_insert(2), Ok(false));
    /// assert_eq!(set.try_insert(16), Err(BitSetError::BiggerThanCapacity));
    /// ```
    #[inline]
    pub fn try_insert(&mut self, bit: usize) -> Result<bool, BitSetError> {
        if bit >= Self::capacity() {
            return Err(BitSetError::BiggerThanCapacity);
        }
        let (index, bitmask) = Self::location(bit);
        Ok(match self.inner.get_mut(index) {
            Some(v) => {
                let contains = *v & bitmask == bitmask;
                // Set the value
                *v = (*v) | bitmask;
                !contains
            },
            None => false,
        })
    }

    /// Inserts a value to the set without making any checks.
    ///
    /// If the set did not have this value present, `true` is returned.
    ///
    /// If the set did have this value present, `false` is returned.
    ///
    /// # Safety
    /// Behavior is undefined if any of the following conditions are violated:
    ///   - The `bit` value is bigger than the capacity of the bitset
    pub unsafe fn insert_unchecked(&mut self, bit: usize) -> bool {
        let (index, bitmask) = Self::location(bit);
        let v = self.inner.get_unchecked_mut(index);
        let contains = *v & bitmask == bitmask;
        *v = (*v) | bitmask;
        !contains
    }

    /// Removes a value from the set. Returns whether the value was present in the set.
    ///
    /// If the bit is already disabled this is a no-op.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbitset::BitSet8;
    ///
    /// let mut set = BitSet8::new();
    ///
    /// set.insert(2);
    /// assert_eq!(set.remove(2), true);
    /// assert_eq!(set.remove(2), false);
    /// ```
    pub fn try_remove(&mut self, bit: usize) -> Result<bool, BitSetError> {
        if bit >= Self::capacity() {
            return Err(BitSetError::BiggerThanCapacity);
        }
        let (index, bitmask) = Self::location(bit);
        Ok(match self.inner.get_mut(index) {
            Some(v) => {
                let was_present = *v & bitmask == bitmask;
                *v = (*v) & !bitmask;
                was_present
            },
            None => false,
        })
    }

    /// Removes a value from the set without any checking. Returns whether the value was present in
    /// the set.
    ///
    /// If the bit is already disabled this is a no-op.
    ///
    /// # Safety
    /// Behavior is undefined if any of the following conditions are violated:
    ///   - The `bit` value is bigger than the capacity of the bitset
    pub unsafe fn remove_unchecked(&mut self, bit: usize) -> bool {
        let (index, bitmask) = Self::location(bit);
        let v = self.inner.get_unchecked_mut(index);
        let was_present = *v & bitmask == bitmask;
        *v = (*v) & !bitmask;
        was_present
    }

    /// Move all elements from `other` into `self`, leaving `other` empty.
    ///
    /// # Panics
    ///
    /// This function may panic if `other` contains activated bits bigger than what `self` capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbitset::BitSet16;
    ///
    /// let mut a = BitSet16::new();
    /// a.insert(1);
    /// a.insert(2);
    /// a.insert(3);
    ///
    /// let mut b = BitSet16::new();
    /// b.insert(3);
    /// b.insert(4);
    /// b.insert(5);
    ///
    /// a.append(&mut b);
    ///
    /// assert_eq!(a.len(), 5);
    /// assert_eq!(b.len(), 0);
    ///
    /// assert!(a.contains(1));
    /// assert!(a.contains(2));
    /// assert!(a.contains(3));
    /// assert!(a.contains(4));
    /// assert!(a.contains(5));
    /// ```
    pub fn append<U, const M: usize>(&mut self, other: &mut BitSet<U, M>)
    where U: PrimInt {
        for item in other.drain() {
            self.insert(item);
        }
    }

    /// Adds a value to the set.
    ///
    /// If the set did not have this value present, `true` is returned.
    ///
    /// If the set did have this value present, `false` is returned.
    ///
    /// # Panics
    /// This function may panic if `bit` value trying to be inserted is bigger than the
    /// [`capacity`](BitSet::capacity) of the [`BitSet`]. Check [`try_insert`](BitSet::try_insert)
    /// for a non-panicking version
    ///
    /// # Examples
    ///
    /// ```
    /// use rbitset::BitSet16;
    ///
    /// let mut set = BitSet16::new();
    ///
    /// assert_eq!(set.insert(2), true);
    /// assert_eq!(set.insert(2), false);
    /// assert_eq!(set.len(), 1);
    /// ```
    #[inline]
    pub fn insert(&mut self, bit: usize) -> bool {
        self.try_insert(bit)
            .expect("BitSet::insert called on an integer bigger than capacity")
    }

    /// Removes a value from the set. Returns whether the value was present in the set.
    ///
    /// If the bit is already disabled this is a no-op.
    ///
    /// # Panics
    /// This function may panic if `bit` value trying to be removed is bigger than the
    /// [`capacity`](BitSet::capacity) of the [`BitSet`]. Check [`try_remove`](BitSet::try_remove)
    /// for a non-panicking version
    ///
    /// # Examples
    ///
    /// ```
    /// use rbitset::BitSet8;
    ///
    /// let mut set = BitSet8::new();
    ///
    /// set.insert(2);
    /// assert_eq!(set.remove(2), true);
    /// assert_eq!(set.remove(2), false);
    /// ```
    pub fn remove(&mut self, bit: usize) -> bool {
        self.try_remove(bit)
            .expect("BitSet::remove called on an integer bigger than capacity")
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` for which `f(&e)` returns `false`.
    /// The elements are visited in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbitset::BitSet16;
    ///
    /// let mut set = BitSet16::from_iter([1u8, 2, 3, 4, 5, 6]);
    /// // Keep only the even numbers.
    /// set.retain(|k| k % 2 == 0);
    /// let res = BitSet16::from_iter([2u8, 4, 6]);
    /// assert_eq!(set, res);
    /// ```
    pub fn retain<F>(&mut self, mut f: F)
    where F: FnMut(usize) -> bool {
        for value in self.clone().iter() {
            if !f(value) {
                // Since we are iteration over the values of the self itself, remove should never
                // panic. Consider to use an unchecked function for removal if the panic code slow
                // it down too much
                self.remove(value);
            }
        }
    }

    /// Returns `true` if the specified `bit` is enabled, in other words, if the set contains a
    /// value.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbitset::BitSet8;
    ///
    /// let set = BitSet8::from_iter([1u8, 2, 3]);
    /// assert_eq!(set.contains(1), true);
    /// assert_eq!(set.contains(4), false);
    /// ```
    pub fn contains(&self, bit: usize) -> bool {
        if bit >= Self::capacity() {
            return false;
        }

        let (index, bitmask) = Self::location(bit);
        match self.inner.get(index) {
            Some(&v) => v & bitmask == bitmask,
            None => false,
        }
    }

    /// Returns `true` if the specified `bit` is enabled, in other words, if the set contains a
    /// value.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbitset::BitSet8;
    ///
    /// let set = BitSet8::from_iter([1u8, 2, 3]);
    /// assert_eq!(set.try_contains(1), Ok(true));
    /// assert_eq!(set.try_contains(4), Ok(false));
    /// ```
    pub fn try_contains(&self, bit: usize) -> Result<bool, BitSetError> {
        if bit >= Self::capacity() {
            return Err(BitSetError::BiggerThanCapacity);
        }

        let (index, bitmask) = Self::location(bit);

        match self.inner.get(index) {
            Some(&v) => Ok(v & bitmask == bitmask),
            None => Err(BitSetError::BiggerThanCapacity),
        }
    }

    /// Returns the number of elements in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbitset::BitSet16;
    ///
    /// let mut set = BitSet16::new();
    /// assert_eq!(set.len(), 0);
    /// set.insert(1);
    /// assert_eq!(set.len(), 1);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.count_ones() as usize
    }

    /// Returns `true` if the set contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbitset::BitSet16;
    ///
    /// let mut set = BitSet16::new();
    /// assert!(set.is_empty());
    /// set.insert(1);
    /// assert!(!set.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `true` if `self` has no elements in common with `other`. This is equivalent to
    /// checking for an empty intersection.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbitset::BitSet128;
    ///
    /// let a = BitSet128::from_iter([1u8, 2, 3]);
    /// let mut b = BitSet128::new();
    ///
    /// assert!(a.is_disjoint(&b));
    /// b.insert(4);
    /// assert!(a.is_disjoint(&b));
    /// b.insert(1);
    /// assert!(!a.is_disjoint(&b));
    /// ```
    pub fn is_disjoint<U: PrimInt, const M: usize>(&self, other: &BitSet<U, M>) -> bool {
        if self.len() <= other.len() {
            self.iter().all(|v| !other.contains(v))
        } else {
            other.iter().all(|v| !self.contains(v))
        }
    }

    /// Returns `true` if the set is a subset of another, i.e., `other` contains at least all the
    /// values in `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbitset::BitSet8;
    ///
    /// let sup = BitSet8::from_iter([1u8, 2, 3]);
    /// let mut set = BitSet8::new();
    ///
    /// assert!(set.is_subset(&sup));
    /// set.insert(2);
    /// assert!(set.is_subset(&sup));
    /// set.insert(4);
    /// assert!(!set.is_subset(&sup));
    /// ```
    pub fn is_subset<U: PrimInt, const M: usize>(&self, other: &BitSet<U, M>) -> bool {
        if self.len() <= other.len() {
            self.iter().all(|v| other.contains(v))
        } else {
            false
        }
    }

    /// Returns `true` if the set is a superset of another, i.e., `self` contains at least all the
    /// values in `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbitset::BitSet8;
    ///
    /// let sub = BitSet8::from_iter([1u8, 2]);
    /// let mut set = BitSet8::new();
    ///
    /// assert!(!set.is_superset(&sub));
    ///
    /// set.insert(0);
    /// set.insert(1);
    /// assert!(!set.is_superset(&sub));
    ///
    /// set.insert(2);
    /// assert!(set.is_superset(&sub));
    /// ```
    #[inline]
    pub fn is_superset<U: PrimInt, const M: usize>(&self, other: &BitSet<U, M>) -> bool {
        other.is_subset(self)
    }

    /// Returns the total number of enabled bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbitset::BitSet8;
    ///
    /// let set = BitSet8::from_iter([1u8, 2, 3]);
    /// assert_eq!(set.count_ones(), 3);
    /// ```
    pub fn count_ones(&self) -> u32 {
        let mut total = 0;
        for item in self.inner.iter() {
            total += item.count_ones();
        }
        total
    }

    /// Returns the total number of disabled bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbitset::BitSet8;
    ///
    /// let set = BitSet8::from_iter([1u8, 2, 3]);
    /// assert_eq!(set.count_zeros(), 5);
    /// ```
    pub fn count_zeros(&self) -> u32 {
        let mut total = 0;

        for item in self.inner.iter() {
            total += item.count_zeros();
        }

        total
    }

    /// Clears the set, returning all elements as an iterator. Keeps the allocated memory for reuse.
    ///
    /// If the returned iterator is dropped before being fully consumed, it drops the remaining
    /// elements. The returned iterator keeps a mutable borrow on the vector to optimize its
    /// implementation.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbitset::BitSet8;
    ///
    /// let mut set = BitSet8::from_iter([1u8, 2, 3]);
    /// assert!(!set.is_empty());
    ///
    /// for i in set.drain() {
    ///     println!("{i}");
    /// }
    ///
    /// assert!(set.is_empty());
    /// ```
    pub fn drain(&mut self) -> Drain<'_, T, N> {
        Drain { inner: self }
    }

    /// Visits the values representing the difference, i.e., the values that are in `self` but not
    /// in `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbitset::BitSet8;
    ///
    /// let a = BitSet8::from_iter([1u8, 2, 3]);
    /// let b = BitSet8::from_iter([4u8, 2, 3, 4]);
    ///
    /// // Can be seen as `a - b`.
    /// for x in a.difference(&b) {
    ///     println!("{x}"); // Print 1
    /// }
    ///
    /// let diff: BitSet8 = a.difference(&b).collect();
    /// let res = BitSet8::from_iter([1u8]);
    /// assert_eq!(diff, res);
    ///
    /// // Note that difference is not symmetric,
    /// // and `b - a` means something else:
    /// let diff: BitSet8 = b.difference(&a).collect();
    /// let res = BitSet8::from_iter([4u8]);
    /// assert_eq!(diff, res);
    /// ```
    pub fn difference<'a, U: PrimInt, const M: usize>(
        &'a self, other: &'a BitSet<U, M>,
    ) -> Difference<'a, T, U, N, M> {
        Difference {
            iter: self.iter(),
            other,
        }
    }

    /// Visits the values representing the intersection, i.e., the values that are both in `self`
    /// and `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbitset::BitSet8;
    ///
    /// let a = BitSet8::from_iter([1u8, 2, 3]);
    /// let b = BitSet8::from_iter([4u8, 2, 3, 4]);
    ///
    /// for x in a.intersection(&b) {
    ///     println!("{x}");
    /// }
    ///
    /// let intersection: BitSet8 = a.intersection(&b).collect();
    /// let test = BitSet8::from_iter([2u8, 3]);
    /// assert_eq!(intersection, test);
    /// ```
    pub fn intersection<'a, U: PrimInt, const M: usize>(
        &'a self, other: &'a BitSet<U, M>,
    ) -> Intersection<'a, T, U, N, M> {
        Intersection {
            iter: self.iter(),
            other,
        }
    }

    /// Visits the values representing the symmetric difference, i.e., the values that are in `self`
    /// or in `other` but not in both.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbitset::BitSet8;
    ///
    /// let a = BitSet8::from_iter([1u8, 2, 3]);
    /// let b = BitSet8::from_iter([4u8, 2, 3, 4]);
    ///
    /// for x in a.symmetric_difference(&b) {
    ///     println!("{x}");
    /// }
    ///
    /// let diff1: BitSet8 = a.symmetric_difference(&b).collect();
    /// let diff2: BitSet8 = b.symmetric_difference(&a).collect();
    ///
    /// assert_eq!(diff1, diff2);
    /// let res = BitSet8::from_iter([1u8, 4]);
    /// assert_eq!(diff1, res);
    /// ```
    pub fn symmetric_difference<'a, U: PrimInt, const M: usize>(
        &'a self, other: &'a BitSet<U, M>,
    ) -> SymmetricDifference<'a, T, U, N, M> {
        SymmetricDifference {
            iter: self.difference(other).chain(other.difference(self)),
        }
    }

    /// Visits the values representing the union, i.e., all the values in `self` or `other`, without
    /// duplicates.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbitset::BitSet8;
    ///
    /// let a = BitSet8::from_iter([1u8, 2, 3]);
    /// let b = BitSet8::from_iter([4u8, 2, 3, 4]);
    ///
    /// for x in a.union(&b) {
    ///     println!("{x}");
    /// }
    ///
    /// let union: BitSet8 = a.union(&b).collect();
    /// let res = BitSet8::from_iter([1u8, 2, 3, 4]);
    /// assert_eq!(union, res);
    /// ```
    pub fn union<'a, U: PrimInt, const M: usize>(
        &'a self, other: &'a BitSet<U, M>,
    ) -> Union<'a, T, U, N, M> {
        if self.len() >= other.len() {
            Union {
                iter: UnionChoose::SelfBiggerThanOther(self.iter().chain(other.difference(self))),
            }
        } else {
            Union {
                iter: UnionChoose::SelfSmallerThanOther(other.iter().chain(self.difference(other))),
            }
        }
    }

    /// An iterator visiting all elements in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use rbitset::BitSet8;
    ///
    /// let mut set = BitSet8::new();
    /// set.insert(1);
    /// set.insert(2);
    ///
    /// for x in set.iter() {
    ///     println!("{x}");
    /// }
    /// ```
    pub fn iter(&self) -> Iter<'_, T, N> {
        Iter::new(self)
    }
}

impl<T: Default + PrimInt, const N: usize> BitSet<T, N> {
    /// Set all bits in a range. `fill(.., false)` is effectively the same as `clear()`.
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
    type Item = usize;
    type IntoIter = IntoIter<T, N>;

    fn into_iter(self) -> Self::IntoIter {
        crate::IntoIter(self)
    }
}

impl<'a, T: PrimInt, const N: usize> IntoIterator for &'a BitSet<T, N> {
    type Item = usize;
    type IntoIter = Iter<'a, T, N>;

    fn into_iter(self) -> Self::IntoIter {
        crate::Iter::new(self)
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

#[cfg(feature = "serde")]
#[cfg_attr(_doc, doc(cfg(feature = "serde")))]
impl<T: PrimInt, const N: usize> Serialize for BitSet<T, N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer {
        let mut seq = serializer.serialize_seq(Some(self.len()))?;
        for ref e in self {
            seq.serialize_element(e)?;
        }
        seq.end()
    }
}

#[cfg(feature = "serde")]
#[cfg_attr(_doc, doc(cfg(feature = "serde")))]
impl<'de, T: PrimInt + Default, const N: usize> Deserialize<'de> for BitSet<T, N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: serde::Deserializer<'de> {
        use core::marker::PhantomData;

        struct BitSetVisitor<T: PrimInt, const N: usize>(PhantomData<BitSet<T, N>>);

        impl<'de, T: PrimInt + Default, const N: usize> Visitor<'de> for BitSetVisitor<T, N> {
            type Value = BitSet<T, N>;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a sequence")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where A: serde::de::SeqAccess<'de> {
                let mut set = BitSet::with_default();

                // While there are entries remaining in the input, add them into our set.
                while let Some(value) = seq.next_element()? {
                    set.insert(value);
                }

                Ok(set)
            }
        }

        let visitor = BitSetVisitor(PhantomData);
        deserializer.deserialize_seq(visitor)
    }
}

/// Possible errors on the [`BitSet`] operations.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[non_exhaustive]
pub enum BitSetError {
    /// Happens when trying to insert or remove a value bigger than the capacity of the bitset.
    BiggerThanCapacity,
}

impl fmt::Display for BitSetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BiggerThanCapacity => f.pad("tried to insert value bigger than capacity"),
        }
    }
}

/// A draining iterator over the items of a `BitSet`.
///
/// This `struct` is created by the [`drain`] method on [`BitSet`]. See its documentation for more.
///
/// [`drain`]: BitSet::drain
///
/// # Examples
///
/// ```
/// use rbitset::BitSet8;
///
/// let mut a = BitSet8::from_iter([1u8, 2, 3]);
///
/// let mut drain = a.drain();
/// ```
pub struct Drain<'a, T: PrimInt + 'a, const N: usize> {
    inner: &'a mut BitSet<T, N>,
}

impl<T: PrimInt, const N: usize> fmt::Debug for Drain<'_, T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.inner, f)
    }
}

impl<'a, T: PrimInt, const N: usize> Iterator for Drain<'a, T, N> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        for (index, item) in self.inner.inner.iter_mut().enumerate() {
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
        let len = self.inner.count_ones() as usize;
        (len, Some(len))
    }
}

impl<T: PrimInt, const N: usize> ExactSizeIterator for Drain<'_, T, N> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<T: PrimInt, const N: usize> FusedIterator for Drain<'_, T, N> {}

/// An owning iterator over the items of a `BitSet`.
///
/// This `struct` is created by the [`into_iter`] method on [`BitSet`] (provided by the
/// [`IntoIterator`] trait). See its documentation for more.
///
/// [`into_iter`]: IntoIterator::into_iter
/// [`IntoIterator`]: core::iter::IntoIterator
///
/// # Examples
///
/// ```
/// use rbitset::BitSet16;
///
/// let a = BitSet16::from_iter([1u8, 2, 3]);
///
/// let mut iter = a.into_iter();
/// ```
#[derive(Clone)]
#[repr(transparent)]
pub struct IntoIter<T, const N: usize>(BitSet<T, N>);

impl<T: PrimInt, const N: usize> fmt::Debug for IntoIter<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_set().entries(self.clone()).finish()
    }
}

impl<T: PrimInt, const N: usize> Iterator for IntoIter<T, N> {
    type Item = usize;

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

/// An iterator over the items of a `BitSet`.
///
/// This `struct` is created by the [`iter`] method on [`BitSet`]. See its documentation for more.
///
/// [`iter`]: BitSet::iter
///
/// # Examples
///
/// ```
/// use rbitset::BitSet8;
///
/// let a = BitSet8::from_iter([1u8, 2, 3]);
///
/// let mut iter = a.iter();
/// ```
#[derive(Clone)]
pub struct Iter<'a, T, const N: usize> {
    borrow: &'a BitSet<T, N>,
    bit: usize,
    passed_count: usize,
}

impl<'a, T: PrimInt, const N: usize> Iter<'a, T, N> {
    fn new(bitset: &'a BitSet<T, N>) -> Self {
        Self {
            borrow: bitset,
            bit: 0,
            passed_count: 0,
        }
    }
}

impl<'a, T: PrimInt, const N: usize> fmt::Debug for Iter<'a, T, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_set().entries(self.clone()).finish()
    }
}

impl<'a, T: PrimInt, const N: usize> Iterator for Iter<'a, T, N> {
    type Item = usize;

    /// Iterator implementation for BitSet, guaranteed to remove and
    /// return the items in ascending order
    fn next(&mut self) -> Option<Self::Item> {
        while !self.borrow.try_contains(self.bit).ok()? {
            self.bit = self.bit.saturating_add(1);
        }

        let res = self.bit;

        self.bit = self.bit.saturating_add(1);
        self.passed_count = self.passed_count.saturating_add(1);

        Some(res)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.borrow.len() - self.passed_count;
        (len, Some(len))
    }
}

impl<'a, T: PrimInt, const N: usize> DoubleEndedIterator for Iter<'a, T, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.bit = self.bit.saturating_sub(2);
        while !self.borrow.try_contains(self.bit).ok()? {
            self.bit = self.bit.saturating_sub(1);
        }

        let res = self.bit;

        self.bit = self.bit.saturating_sub(1);
        self.passed_count = self.passed_count.saturating_sub(1);

        Some(res)
    }
}

impl<'a, T: PrimInt, const N: usize> FusedIterator for Iter<'a, T, N> {}
impl<'a, T: PrimInt, const N: usize> ExactSizeIterator for Iter<'a, T, N> {}

/// A lazy iterator producing elements in the difference of `BitSet`s.
///
/// This `struct` is created by the [`difference`] method on [`BitSet`]. See its documentation for
/// more.
///
/// [`difference`]: BitSet::difference
///
/// # Examples
///
/// ```
/// use rbitset::BitSet8;
///
/// let a = BitSet8::from_iter([1u8, 2, 3]);
/// let b = BitSet8::from_iter([4u8, 2, 3, 4]);
///
/// let mut difference = a.difference(&b);
/// ```
#[must_use = "this returns the difference as an iterator, without modifying either input set"]
#[derive(Clone)]
pub struct Difference<'a, T: PrimInt + 'a, U: PrimInt + 'a, const N: usize, const M: usize> {
    // iterator of the first set
    iter:  Iter<'a, T, N>,
    // the second set
    other: &'a BitSet<U, M>,
}

impl<'a, T, U, const N: usize, const M: usize> fmt::Debug for Difference<'a, T, U, N, M>
where
    T: PrimInt,
    U: PrimInt,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_set().entries(self.clone()).finish()
    }
}

impl<'a, T, U, const N: usize, const M: usize> Iterator for Difference<'a, T, U, N, M>
where
    T: PrimInt + 'a,
    U: PrimInt + 'a,
{
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let elt = self.iter.next()?;
            if !self.other.contains(elt) {
                return Some(elt);
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper)
    }
}

impl<T, U, const N: usize, const M: usize> FusedIterator for Difference<'_, T, U, N, M>
where
    T: PrimInt,
    U: PrimInt,
{
}

/// A lazy iterator producing elements in the intersection of `BitSet`s.
///
/// This `struct` is created by the [`intersection`] method on [`BitSet`]. See its documentation for
/// more.
///
/// [`intersection`]: BitSet::intersection
///
/// # Examples
///
/// ```
/// use rbitset::BitSet8;
///
/// let a = BitSet8::from_iter([1u8, 2, 3]);
/// let b = BitSet8::from_iter([4u8, 2, 3, 4]);
///
/// let mut intersection = a.intersection(&b);
/// ```
#[must_use = "this returns the intersection as an iterator, without modifying either input set"]
#[derive(Clone)]
pub struct Intersection<'a, T, U, const N: usize, const M: usize>
where
    T: PrimInt + 'a,
    U: PrimInt + 'a,
{
    // iterator of the first set
    iter:  Iter<'a, T, N>,
    // the second set
    other: &'a BitSet<U, M>,
}

impl<'a, T, U, const N: usize, const M: usize> fmt::Debug for Intersection<'a, T, U, N, M>
where
    T: PrimInt,
    U: PrimInt,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_set().entries(self.clone()).finish()
    }
}

impl<'a, T, U, const N: usize, const M: usize> Iterator for Intersection<'a, T, U, N, M>
where
    T: PrimInt + 'a,
    U: PrimInt + 'a,
{
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let elt = self.iter.next()?;
            if self.other.contains(elt) {
                return Some(elt);
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper)
    }
}

impl<T, U, const N: usize, const M: usize> FusedIterator for Intersection<'_, T, U, N, M>
where
    T: PrimInt,
    U: PrimInt,
{
}

/// A lazy iterator producing elements in the union of `BitSet`s.
///
/// This `struct` is created by the [`union`] method on [`BitSet`]. See its documentation for more.
///
/// [`union`]: BitSet::union
///
/// # Examples
///
/// ```
/// use rbitset::BitSet8;
///
/// let a = BitSet8::from_iter([1u8, 2, 3]);
/// let b = BitSet8::from_iter([4u8, 2, 3, 4]);
///
/// let mut union_iter = a.union(&b);
/// ```
#[must_use = "this returns the union as an iterator, without modifying either input set"]
#[derive(Clone)]
pub struct Union<'a, T: PrimInt + 'a, U: PrimInt + 'a, const N: usize, const M: usize> {
    iter: UnionChoose<'a, T, U, N, M>,
}

impl<'a, T, U, const N: usize, const M: usize> fmt::Debug for Union<'a, T, U, N, M>
where
    T: PrimInt,
    U: PrimInt,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_set().entries(self.clone()).finish()
    }
}

impl<'a, T, U, const N: usize, const M: usize> Iterator for Union<'a, T, U, N, M>
where
    T: PrimInt + 'a,
    U: PrimInt + 'a,
{
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<T, U, const N: usize, const M: usize> FusedIterator for Union<'_, T, U, N, M>
where
    T: PrimInt,
    U: PrimInt,
{
}

/// Helper enum to create a [`Union`] for [`BitSet`]
#[derive(Clone)]
enum UnionChoose<'a, T: PrimInt, U: PrimInt, const N: usize, const M: usize> {
    SelfBiggerThanOther(Chain<Iter<'a, T, N>, Difference<'a, U, T, M, N>>),
    SelfSmallerThanOther(Chain<Iter<'a, U, M>, Difference<'a, T, U, N, M>>),
}

impl<'a, T, U, const N: usize, const M: usize> Iterator for UnionChoose<'a, T, U, N, M>
where
    T: PrimInt + 'a,
    U: PrimInt + 'a,
{
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::SelfBiggerThanOther(iter) => iter.next(),
            Self::SelfSmallerThanOther(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::SelfBiggerThanOther(iter) => iter.size_hint(),
            Self::SelfSmallerThanOther(iter) => iter.size_hint(),
        }
    }
}

impl<T, U, const N: usize, const M: usize> FusedIterator for UnionChoose<'_, T, U, N, M>
where
    T: PrimInt,
    U: PrimInt,
{
}

/// A lazy iterator producing elements in the symmetric difference of `BitSet`s.
///
/// This `struct` is created by the [`symmetric_difference`] method on [`BitSet`]. See its
/// documentation for more.
///
/// [`symmetric_difference`]: BitSet::symmetric_difference
///
/// # Examples
///
/// ```
/// use rbitset::BitSet8;
///
/// let a = BitSet8::from_iter([1u8, 2, 3]);
/// let b = BitSet8::from_iter([4u8, 2, 3, 4]);
///
/// let mut intersection = a.symmetric_difference(&b);
/// ```
#[must_use = "this returns the difference as an iterator, without modifying either input set"]
#[derive(Clone)]
pub struct SymmetricDifference<'a, T, U, const N: usize, const M: usize>
where
    T: PrimInt + 'a,
    U: PrimInt + 'a,
{
    iter: Chain<Difference<'a, T, U, N, M>, Difference<'a, U, T, M, N>>,
}

impl<'a, T, U, const N: usize, const M: usize> fmt::Debug for SymmetricDifference<'a, T, U, N, M>
where
    T: PrimInt,
    U: PrimInt,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_set().entries(self.clone()).finish()
    }
}

impl<'a, T, U, const N: usize, const M: usize> Iterator for SymmetricDifference<'a, T, U, N, M>
where
    T: PrimInt + 'a,
    U: PrimInt + 'a,
{
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<T, U, const N: usize, const M: usize> FusedIterator for SymmetricDifference<'_, T, U, N, M>
where
    T: PrimInt,
    U: PrimInt,
{
}

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
    fn contains() {
        let mut b = BitSet8::new();
        b.insert(0);
        b.insert(1);
        b.insert(2);
        b.insert(3);
        b.insert(4);
        b.insert(5);
        b.insert(6);
        b.insert(7);
        assert!(b.contains(0));
        assert!(b.contains(1));
        assert!(b.contains(2));
        assert!(b.contains(3));
        assert!(b.contains(4));
        assert!(b.contains(5));
        assert!(b.contains(6));
        assert!(b.contains(7));
        assert!(!b.contains(8));
        assert!(!b.contains(9));
        assert!(!b.contains(10));
        assert!(!b.contains(11));
        assert!(!b.contains(12));
        assert!(!b.contains(13));
        assert!(!b.contains(14));
        assert!(!b.contains(15));
    }

    #[test]
    fn try_too_big() {
        let mut set = BitSet8::new();
        assert_eq!(set.try_insert(8), Err(BitSetError::BiggerThanCapacity));
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
        assert!(set.insert(127));
        assert!(!set.insert(127));

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
    fn insert_unchecked() {
        let mut set = BitSet128::new();
        unsafe {
            set.insert_unchecked(0);
            set.insert_unchecked(12);
            set.insert_unchecked(67);
            set.insert_unchecked(82);
            assert!(set.insert_unchecked(127));
            assert!(!set.insert_unchecked(127));
        }

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
    fn remove_unchecked() {
        let mut set = BitSet32::new();
        set.insert(12);
        set.insert(17);
        assert!(set.contains(12));
        assert!(set.contains(17));
        unsafe { set.remove_unchecked(17) };
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
    fn retain() {
        let mut set = BitSet16::from_iter([1u8, 2, 3, 4, 5, 6]);
        // Keep only the even numbers.
        set.retain(|k| k % 2 == 0);
        let res = BitSet16::from_iter([2u8, 4, 6]);
        assert_eq!(set, res);
    }

    #[test]
    fn append() {
        let mut a = BitSet16::new();
        a.insert(1);
        a.insert(2);
        a.insert(3);

        let mut b = BitSet16::new();
        b.insert(3);
        b.insert(4);
        b.insert(5);

        a.append(&mut b);

        assert_eq!(a.len(), 5);
        assert_eq!(b.len(), 0);

        assert!(a.contains(1));
        assert!(a.contains(2));
        assert!(a.contains(3));
        assert!(a.contains(4));
        assert!(a.contains(5));
    }

    #[test]
    fn disjoint() {
        let a = BitSet128::from_iter([1u8, 2, 3]);
        let mut b = BitSet128::new();

        assert!(a.is_disjoint(&b));
        assert!(b.is_disjoint(&a));
        b.insert(4);
        assert!(a.is_disjoint(&b));
        assert!(b.is_disjoint(&a));
        b.insert(1);
        assert!(!a.is_disjoint(&b));
        assert!(!b.is_disjoint(&a));
    }

    #[test]
    fn subset_superset() {
        let sup = BitSet8::from_iter([1u8, 2, 3]);
        let mut set = BitSet8::new();

        // A superset is never a subset of it's subsets and vice versa
        assert!(!sup.is_subset(&set));
        assert!(!set.is_superset(&sup));
        assert!(set.is_subset(&sup));
        assert!(sup.is_superset(&set));
        set.insert(2);
        // A superset is never a subset of it's subsets
        assert!(!sup.is_subset(&set));
        assert!(!set.is_superset(&sup));
        assert!(set.is_subset(&sup));
        assert!(sup.is_superset(&set));
        set.insert(4);
        assert!(!set.is_subset(&sup));
        assert!(!sup.is_superset(&set));
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
    fn drain() {
        let mut set = BitSet8::from_iter([1u8, 2, 3]);
        assert!(!set.is_empty());

        for _ in set.drain() {}

        assert!(set.is_empty());
    }

    #[test]
    fn iter() {
        let set: BitSet<u8, 4> = [30u8, 0, 4, 2, 12, 22, 23, 29].iter().copied().collect();

        // Back and forth
        let mut iter = set.iter();
        assert_eq!(iter.len(), 8);
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.len(), 7);
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.len(), 6);
        assert_eq!(iter.next_back(), Some(0));
        assert_eq!(iter.len(), 7);

        // One way
        let mut iter = set.iter();
        assert_eq!(iter.len(), 8);
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.len(), 7);
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.len(), 6);
        assert_eq!(iter.next(), Some(4));
        assert_eq!(iter.len(), 5);
        assert_eq!(iter.next(), Some(12));
        assert_eq!(iter.len(), 4);
        assert_eq!(iter.next(), Some(22));
        assert_eq!(iter.len(), 3);
        assert_eq!(iter.next(), Some(23));
        assert_eq!(iter.len(), 2);
        assert_eq!(iter.next(), Some(29));
        assert_eq!(iter.len(), 1);
        assert_eq!(iter.next(), Some(30));
        assert_eq!(iter.len(), 0);
        assert_eq!(iter.next(), None);
        assert_eq!(iter.len(), 0);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn into_iter() {
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
    fn difference() {
        let a = BitSet8::from_iter([1u8, 2, 3]);
        let b = BitSet8::from_iter([4u8, 2, 3, 4]);

        let diff: BitSet8 = a.difference(&b).collect();
        let res = BitSet8::from_iter([1u8]);
        assert_eq!(diff, res);

        // Note that difference is not symmetric,
        // and `b - a` means something else:
        let diff: BitSet8 = b.difference(&a).collect();
        let res = BitSet8::from_iter([4u8]);
        assert_eq!(diff, res);
    }

    #[test]
    fn intersect() {
        let a = BitSet8::from_iter([1u8, 2, 3]);
        let b = BitSet8::from_iter([4u8, 2, 3, 4]);

        let intersection: BitSet8 = a.intersection(&b).collect();
        let test = BitSet8::from_iter([2u8, 3]);
        assert_eq!(intersection, test);
    }

    #[test]
    fn union() {
        let a = BitSet8::from_iter([1u8, 2, 3]);
        let b = BitSet8::from_iter([4u8, 2, 3, 4]);

        let union: BitSet8 = a.union(&b).collect();
        let res = BitSet8::from_iter([1u8, 2, 3, 4]);
        assert_eq!(union, res);

        let a = BitSet8::from_iter([1u8, 2, 3]);
        let b = BitSet8::from_iter([4u8, 2, 3, 4, 5]);
        let union: BitSet8 = a.union(&b).collect();
        let res = BitSet8::from_iter([1u8, 2, 3, 4, 5]);
        assert_eq!(union, res);
    }

    #[test]
    fn symmetric_difference() {
        let a = BitSet8::from_iter([1u8, 2, 3]);
        let b = BitSet8::from_iter([4u8, 2, 3, 4]);

        let diff1: BitSet8 = a.symmetric_difference(&b).collect();
        let diff2: BitSet8 = b.symmetric_difference(&a).collect();

        assert_eq!(diff1, diff2);
        let res = BitSet8::from_iter([1u8, 4]);
        assert_eq!(diff1, res);
    }

    #[test]
    fn debug() {
        use self::alloc::format;
        assert_eq!(
            format!("{:?}", (0u16..10).collect::<BitSet16>()),
            "{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}"
        );
        assert_eq!(
            format!("{:#?}", (0u16..10).collect::<BitSet16>()),
            "{\n    0,\n    1,\n    2,\n    3,\n    4,\n    5,\n    6,\n    7,\n    8,\n    9,\n}"
        );
    }

    #[test]
    fn binary() {
        use self::alloc::format;
        assert_eq!(
            format!("{:b}", (0u16..10).collect::<BitSet16>()),
            "BitSet [0b0000001111111111]"
        );
        assert_eq!(
            format!("{:#b}", (0u16..10).collect::<BitSet16>()),
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

    #[test]
    fn extend() {
        let mut set = BitSet16::new();
        assert!(set.is_empty());

        set.extend(0..10_usize);
        assert_eq!(set.len(), 10);
        assert_eq!(set, BitSet16::from_iter(0..10_usize));
    }
}

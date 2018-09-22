#![no_std]

extern crate num_traits;

use core::mem;
use num_traits::{PrimInt, FromPrimitive};

pub trait BitArray: Default + Clone + Copy {
    /// The item type this array holds
    type Item: Default + PrimInt + FromPrimitive;
    /// Returns how many elements this array can hold
    fn len() -> usize;
    /// Access the element at a specified index.
    ///
    /// # Panics
    /// Panics if the index is bigger than the length
    fn get(&self, index: usize) -> Self::Item;
    /// Access a mutable reference to the element at a specified index
    ///
    /// # Panics
    /// Panics if the index is bigger than the length
    fn get_mut(&mut self, index: usize) -> &mut Self::Item;
}

macro_rules! impl_arrays {
    ($($len:expr),*) => {
        $(
            impl<T: Default + PrimInt + FromPrimitive> BitArray for [T; $len] {
                type Item = T;

                fn len() -> usize { $len }
                fn get(&self, index: usize) -> Self::Item {
                    self[index]
                }
                fn get_mut(&mut self, index: usize) -> &mut Self::Item {
                    &mut self[index]
                }
            }
        )*
    }
}

impl_arrays!(1, 2, 3, 4, 5, 6, 7, 8);

/// A bit set able to hold up to 8 elements
pub type BitSet8 = BitSet<[u8; 1]>;
/// A bit set able to hold up to 16 elements
pub type BitSet16 = BitSet<[u16; 1]>;
/// A bit set able to hold up to 32 elements
pub type BitSet32 = BitSet<[u32; 1]>;
/// A bit set able to hold up to 64 elements
pub type BitSet64 = BitSet<[u64; 1]>;
/// A bit set able to hold up to 128 elements
pub type BitSet128 = BitSet<[u64; 2]>;
/// A bit set able to hold up to 256 elements
pub type BitSet256 = BitSet<[u64; 4]>;
/// A bit set able to hold up to 512 elements
pub type BitSet512 = BitSet<[u64; 8]>;

/// The bit set itself
///
/// This wrapper is `#![repr(transparent)]` and guaranteed to have the same memory
/// representation as the inner bit array
///
/// # Panics
/// All non-try functions taking a bit parameter panics if the bit is bigger
/// than the capacity of the set. For non-panicking versions, use `try_`.
#[repr(transparent)]
#[derive(Default, Clone, Copy)]
pub struct BitSet<T: BitArray> {
    inner: T
}
impl<T: BitArray> BitSet<T> {
    /// Create an empty instance
    pub fn new() -> Self {
        Self::default()
    }
    /// Returns the capacity of the set, in other words how many bits it can
    /// hold. This may very well overflow if the size or length is too big, but
    /// if you're making that big allocations you probably got bigger things to
    /// worry about.
    pub fn capacity() -> usize {
        T::len() * Self::item_size()
    }
    /// Returns the bit size of each item
    fn item_size() -> usize {
        mem::size_of::<T::Item>() * 8
    }
    /// Converts a u64 to a T::Item
    fn from(i: u64) -> T::Item {
        FromPrimitive::from_u64(i).unwrap()
    }

    /// Empty the set, clearing all bits
    pub fn clear(&mut self) {
        for i in 0..T::len() {
            *self.inner.get_mut(i) = Default::default();
        }
    }
    /// Enable the value `bit` in the set.
    /// If `bit` is already enabled this is a no-op.
    pub fn insert(&mut self, bit: usize) {
        assert!(self.try_insert(bit), "BitSet::insert called on an integer bigger than capacity");
    }
    /// Like insert, but does not panic if the bit is too large.
    /// See the struct level documentation for notes on panicking.
    pub fn try_insert(&mut self, bit: usize) -> bool {
        if bit >= Self::capacity() {
            return false
        }
        let index = bit / Self::item_size();
        *self.inner.get_mut(index) =
            self.inner.get(index) | Self::from(1 << (bit & Self::item_size() - 1));
        true
    }
    /// Disable the value `bit` in the set.
    /// If `bit` is already disabled this is a no-op.
    pub fn remove(&mut self, bit: usize) {
        assert!(self.try_remove(bit), "BitSet::remove called on an integer bigger than capacity");
    }
    /// Like remove, but does not panic if the bit is too large.
    /// See the struct level documentation for notes on panicking.
    pub fn try_remove(&mut self, bit: usize) -> bool {
        if bit >= Self::capacity() {
            return false
        }
        let index = bit / Self::item_size();
        *self.inner.get_mut(index) =
            self.inner.get(index) & Self::from(!(1 << (bit & Self::item_size() - 1)));
        true
    }
    /// Returns true if the bit `bit` is enabled.
    /// If the bit is out of bounds this silently returns false.
    pub fn contains(&self, bit: usize) -> bool {
        self.try_contains(bit).unwrap_or(false)
    }
    /// Returns true if the bit `bit` is enabled.
    pub fn try_contains(&self, bit: usize) -> Option<bool> {
        if bit >= Self::capacity() {
            return None;
        }

        let mask = Self::from(1 << (bit & Self::item_size() - 1));
        Some(self.inner.get(bit / Self::item_size()) & mask == mask)
    }

    /// Returns the total number of enabled bits
    pub fn count_ones(&self) -> u32 {
        let mut total = 0;
        for i in 0..T::len() {
            total += self.inner.get(i).count_ones();
        }
        total
    }
    /// Returns the total number of disabled bits
    pub fn count_zeros(&self) -> u32 {
        let mut total = 0;
        for i in 0..T::len() {
            total += self.inner.get(i).count_zeros();
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut set = BitSet64::new();
        set.insert(35);
        set.insert(42);
        assert!(set.contains(35));
        assert!(set.contains(42));
        set.remove(42);
        assert!(set.contains(35));
        assert!(!set.contains(42));
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
}

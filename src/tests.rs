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

extern crate cbitset;

use cbitset::BitSet256;

fn main() {
    println!("The first {} bytes of \"hello\" are in \"eh\"",   strspn(b"hello", b"eh"));
    println!("The first {} bytes of \"hello\" are in \"ehlo\"", strspn(b"hello", b"ehlo"));
    println!("The first {} bytes of \"it works\" are letters",  strspn(b"it works", b"abcdefghijklmnopqrstuwxyz"));
}

/// The C standard library function strspn, reimplemented in rust. It works by
/// placing all allowed values in a bit set, and returning on the first
/// character not on the list. A BitSet256 uses no heap allocations and only 4
/// 64-bit integers in stack memory.
fn strspn(s: &[u8], accept: &[u8]) -> usize {
    let mut allow = BitSet256::new();

    for &c in accept {
        allow.insert(c as usize);
    }

    for (i, &c) in s.iter().enumerate() {
        if !allow.contains(c as usize) {
            return i;
        }
    }
    s.len()
}

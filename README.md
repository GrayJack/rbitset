# cbitset [![Crates.io](https://img.shields.io/crates/v/cbitset.svg)](https://crates.io/crates/cbitset)

A bit set, being able to hold a fixed amount of booleans in an array of
integers.

## Alternatives

There are already quite a few libraries out there for bit sets, but I can't
seem to find a `#![no_std]` one that works with fixed-sized arrays. Most of
them seem to want to be dynamic.

cbitset also is `repr(transparent)`, meaning the representation of the struct
is guaranteed to be the same as the inner array, making it usable from stuff
where the struct representation is important, such as
[relibc](https://gitlab.redox-os.org/redox-os/relibc).

## Inspiration

I think this is a relatively common thing to do in C, for I stumbled upon the
concept in the [MUSL](https://www.musl-libc.org/) standard library. An example
is its usage in
[strspn](https://git.musl-libc.org/cgit/musl/tree/src/string/strspn.c).

While it's a relatively easy concept, the implementation can be pretty
unreadable. So maybe it should be abstracted away with some kind of... zero
cost abstraction?

## Example

Bit sets are extremely cheap. You can store any number from 0 to 255 in an
array of 4x 64-bit numbers. Lookup should in theory be O(1). An example usage
of this is once again `strspn`. Here it is in rust, using this library:
```rust
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
```

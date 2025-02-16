# Changelog

All notable changes to this project will be documented in this file. See
[conventional commits](https://www.conventionalcommits.org/) for commit
guidelines.

---

## v0.3.3 - 2025-02-02

### Bug Fixes

- Fix clippy lints - (71c3087) - GrayJack
- Fix warnings - (524ad20) - GrayJack

### Continuous Integration

- Update actions - (cd693df) - GrayJack
- Move to checkout v3 on docs and fmt actions - (297c5b3) - GrayJack
- Avoid running CI with changes on files not related to code - (5152ccf) - GrayJack
- Simplify `build` workflow - (cbb950b) - GrayJack
- Use `dtolnay/rust-toolchain` for the rust toolchain instead - (24e514a) - GrayJack
- Add coverage workflow - (7f3e528) - GrayJack

### Documentation

- Add module level docs - (47d1282) - GrayJack
- Use the `docsrs` cfg flag - (9101baa) - GrayJack

### Features

- Add `must_use` on all iterators - (4f0eff3) - GrayJack

### Miscellaneous Chores

- Update cog.toml - (e084a59) - GrayJack
- bump version - (f7be11b) - GrayJack
- Update README - (ae7c6c5) - GrayJack
- Bump version - (4ba2e9a) - GrayJack
- Simplify - (da926f2) - GrayJack
- Fix typos - (5565172) - GrayJack
- Add .deepsource.toml - (c51e196) - GrayJack
- Activate Github sponsors - (fc90779) - GrayJack

### Performance Improvements

- Specialize `Iterator::count` when for `Drain`, `IntoIter` and `Iter` - (5e5f1ba) - GrayJack
- Specialize `ExactSizeIterator::len` when possible - (0726d88) - GrayJack

### Tests

- Improve test coverage - (c2f8342) - GrayJack

---

## v0.3.1 - 2022-06-07

### Features

- Add unchecked version of insert and remove operations - (2d9a9aa) - GrayJack

---

## v0.3.0 - 2022-05-29

### Documentation

- Fix example in the docs for `try_append` and `append` - (437f7b5) - GrayJack

### Features

- Add `retain` method for `BitSet` - (63238ad) - GrayJack
- Add `{try_}append` API - (c857f29) - GrayJack

### Miscellaneous Chores

- update gitignore - (69b75d2) - GrayJack

---

## v0.2.1 - 2022-05-19

### Features

- Add optional serde support - (dc58177) - GrayJack

### Miscellaneous Chores

- Update `Cargo.toml` values - (2b8225d) - GrayJack

---

## v0.2.0 - 2022-05-19

### Documentation

- Add examples to all functions part of the public API and reword a few items - (8dc0383) - GrayJack
- Improve documentation for `IntoIter` type - (0b35df3) - GrayJack
- Improve documentation for `Iter` related items - (68e0e77) - GrayJack
- Improve parts of the documentation - (c5589af) - GrayJack

### Features

- Introduce a Drain iterator - (478a33a) - GrayJack
- Change the `remove` and `contains` to match the semantics of standard library HashSet and BtreeSet - (a24c3cc) - GrayJack
- Change the insert to match the semantics of standard library `HashSet` and `BtreeSet` - (afd79f9) - GrayJack
- Rework again the debug implementation and move old implementation to `fmt::Binary` formatting - (c4c0eb5) - GrayJack
- Changed the debug printing of `BitSet`. - (5045e2d) - GrayJack
- Make `Debug` formatting for `Iter` and `IntoIter` similar to `HashSet`/`BtreeSet` - (07a4f94) - GrayJack
- Implement iterator over symetric difference between two `BitSet`s - (a024487) - GrayJack
- Implement iterator over the union between two `BitSet`s - (36024d1) - GrayJack
- Implement iterator over the difference between two `BitSet`s - (3265c35) - GrayJack
- Implement iterator over intersection over 2 `BitSet`s - (4f0ff4c) - GrayJack
- Add functions to check relationships between bitsets (`disjoint`, `subset` and `superset`) - (a7ec18e) - GrayJack
- Create a non-consuming iterator for `BitSet` - (aa3dd65) - GrayJack
- Add `len` and `is_empty` to `BitSet` - (07adfa7) - GrayJack
- Implement `Extend` for `BitSet` - (98a1f0f) - GrayJack
- Implement `Extend` for `BitSet` - (752bede) - GrayJack

### Miscellaneous Chores

- Update the changelog template - (bf296e7) - GrayJack
- Fix typo on README.md - (1c98065) - GrayJack
- Update README.md and CHANGELOG.md - (ee4e8aa) - GrayJack

### Refactoring

- Create a `IntoIter` type for `BitSet` - (5d8ed79) - GrayJack

---

## v0.1.0 - 2022-03-05

### Continuous Integration

- Fix ctrl+c ctrl+v problem - (f5fec17) - GrayJack
- Configure Github Actions - (9de18c1) - GrayJack

### Features

- Implement binary not - (d2331de) - jD91mZM2
- Implement Debug - (ea610ef) - jD91mZM2
- Implement iterator for BitSet - (84a4b20) - jD91mZM2
- Add safe ways to convert back and forth from underlying array - (6807e88) -
  jD91mZM2
- Implement a fill() function - (56dfd7f) - jD91mZM2

### Miscellaneous Chores

- **version:** 0.1.0 - (950e007) - GrayJack
- Tweak conventional commit configuration - (51afa91) - GrayJack
- Update README - (2877aa3) - GrayJack
- Add rustfmt configuration - (411e537) - GrayJack
- raname to rbitset - (f2f02c5) - GrayJack
- Configure conventional commits - (a8b054e) - GrayJack
- Delete Cargo.lock - (f68ed2f) - jD91mZM2
- Push to crates.io - (7b07f7d) - jD91mZM2

### Refactoring

- Reimplement `BitSet` type based on const generics - (c5fb1ff) - GrayJack
- Apply clippy fixes - (773eecd) - GrayJack

---

Changelog generated by [cocogitto](https://github.com/cocogitto/cocogitto).

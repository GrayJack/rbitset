# Changelog

All notable changes to this project will be documented in this file.

## [0.3.5](https://github.com/GrayJack/rbitset/compare/v0.3.4..v0.3.5) — 2025-02-16

### 📚 Documentation

- Update and cleanup doc attributes — ([f8f6bbb](https://github.com/GrayJack/rbitset/commit/f8f6bbb430c5404f8281b07a103d5f2b950e775a))
- Add changelog to rustdoc — ([1a5ce4f](https://github.com/GrayJack/rbitset/commit/1a5ce4f7b5cae43e89dbc88aa207df81cc59ee63))
- Improve entry documentation — ([3c4f03f](https://github.com/GrayJack/rbitset/commit/3c4f03f0bd4825a8879fc0fcb9d955942303673f))
- Improve `BitSet` type documentation — ([8e142be](https://github.com/GrayJack/rbitset/commit/8e142be3f9af606206deb63ae42aa4c5710d40c1))
- Fix typos — ([1346c48](https://github.com/GrayJack/rbitset/commit/1346c48035dab8da0bd56e723f6df380c30df23a))

### ⚙️ Continuous Integration

- Fix doc generation — ([d26eacf](https://github.com/GrayJack/rbitset/commit/d26eacf2af3a79c7b75ae762a0bf4bffb78b7738))

## [0.3.4](https://github.com/GrayJack/rbitset/compare/v0.3.3..v0.3.4) — 2025-02-16

### 🚜 Refactor

- Move tests module to a file — ([8623094](https://github.com/GrayJack/rbitset/commit/8623094391b944176c0229a15da72daad8710891))

## [0.3.3](https://github.com/GrayJack/rbitset/compare/v0.3.1..v0.3.3) — 2025-02-02

### 🚀 Features

- Add `must_use` on all iterators — ([4f0eff3](https://github.com/GrayJack/rbitset/commit/4f0eff32cf51ea4d2399e25953500331a7ee09d3))

### 🐞 Bug Fixes

- Fix warnings — ([524ad20](https://github.com/GrayJack/rbitset/commit/524ad201e213f0bd089f29ac983d09e7817e48f6))
- Fix clippy lints — ([71c3087](https://github.com/GrayJack/rbitset/commit/71c3087443db1e0c039ba48b9b1c580979603e6c))

### 📚 Documentation

- Use the `docsrs` cfg flag — ([9101baa](https://github.com/GrayJack/rbitset/commit/9101baa9e8b33aabcc945ced9ba88217c5f6eb12))
- Add module level docs — ([47d1282](https://github.com/GrayJack/rbitset/commit/47d12829e29c3ac3a0da61edc698b49e3ac9713f))

### ⚡ Performance

- Specialize `ExactSizeIterator::len` when possible — ([0726d88](https://github.com/GrayJack/rbitset/commit/0726d883b784fe6bf3d9bc8c8d1ae5f79c751c98))
- Specialize `Iterator::count` when for `Drain`, `IntoIter` and `Iter` — ([5e5f1ba](https://github.com/GrayJack/rbitset/commit/5e5f1bafc4db3ce9ed02233b5ff9179de8f33ac1))

### 🧪 Testing

- Improve test coverage — ([c2f8342](https://github.com/GrayJack/rbitset/commit/c2f8342628515382babb7b53ca61acf8959d15da))

### ⚙️ Continuous Integration

- Add coverage workflow — ([7f3e528](https://github.com/GrayJack/rbitset/commit/7f3e5285fe2bcede7ba4deb7aedb1b56df903243))
- Use `dtolnay/rust-toolchain` for the rust toolchain instead — ([24e514a](https://github.com/GrayJack/rbitset/commit/24e514af98fcfce9e5e1ea5a159aa8f642baf923))
- Simplify `build` workflow — ([cbb950b](https://github.com/GrayJack/rbitset/commit/cbb950b5ad289b3e39ca6b146fed10e0441bd359))
- Avoid running CI with changes on files not related to code — ([5152ccf](https://github.com/GrayJack/rbitset/commit/5152ccf6582902205ce4f2967df619079e57d9af))
- Move to checkout v3 on docs and fmt actions — ([297c5b3](https://github.com/GrayJack/rbitset/commit/297c5b3924039935cdcf291a538eabf1235e915c))
- Update actions — ([cd693df](https://github.com/GrayJack/rbitset/commit/cd693dfef3b92310f558377c1a0437ead2e59a1d))

## [0.3.1](https://github.com/GrayJack/rbitset/compare/v0.3.0..v0.3.1) — 2022-06-07

### 🚀 Features

- Add unchecked version of insert and remove operations — ([2d9a9aa](https://github.com/GrayJack/rbitset/commit/2d9a9aae2ff6e44f73bd79e55a9a93a4031fb2af))

## [0.3.0](https://github.com/GrayJack/rbitset/compare/v0.2.1..v0.3.0) — 2022-05-29

### 🚀 Features

- Add `{try_}append` API — ([c857f29](https://github.com/GrayJack/rbitset/commit/c857f29c50c6f69ad5608e4f6e84116eb8e9ca4c))
- Add `retain` method for `BitSet` — ([63238ad](https://github.com/GrayJack/rbitset/commit/63238adac38761815123975a865ecc06ac29d920))

### 📚 Documentation

- Fix example in the docs for `try_append` and `append` — ([437f7b5](https://github.com/GrayJack/rbitset/commit/437f7b5b6d9de5ec229dd066736ce6d8c5134476))

## [0.2.1](https://github.com/GrayJack/rbitset/compare/v0.2.0..v0.2.1) — 2022-05-19

### 🚀 Features

- Add optional serde support — ([dc58177](https://github.com/GrayJack/rbitset/commit/dc58177d4b33184dd54c87a6e1b059d423fb18f4))

## [0.2.0](https://github.com/GrayJack/rbitset/compare/v0.1.0..v0.2.0) — 2022-05-19

### 🚀 Features

- Implement `Extend` for `BitSet` — ([752bede](https://github.com/GrayJack/rbitset/commit/752bede584156a10f47d7003f6c0e427238872bd))
- (**BREAKING**) Implement `Extend` for `BitSet` — ([98a1f0f](https://github.com/GrayJack/rbitset/commit/98a1f0fcc5cd2adff62ac7146e31e95c19a35ced))
- Add `len` and `is_empty` to `BitSet` — ([07adfa7](https://github.com/GrayJack/rbitset/commit/07adfa70dab9baaf660ff73cacb5dd3addef0138))
- Create a non-consuming iterator for `BitSet` — ([aa3dd65](https://github.com/GrayJack/rbitset/commit/aa3dd653ae35f81f2e10d5a2715975320592e985))
- Add functions to check relationships between bitsets (`disjoint`, `subset` and `superset`) — ([a7ec18e](https://github.com/GrayJack/rbitset/commit/a7ec18eea963986ebd20bcd6584fa31ec1be5e3c))
- Implement iterator over intersection over 2 `BitSet`s — ([4f0ff4c](https://github.com/GrayJack/rbitset/commit/4f0ff4c0a12804bde283263482e9d9c7f7585b6b))
- Implement iterator over the difference between two `BitSet`s — ([3265c35](https://github.com/GrayJack/rbitset/commit/3265c358766a73e72d76f129a53a661080d2bce4))
- Implement iterator over the union between two `BitSet`s — ([36024d1](https://github.com/GrayJack/rbitset/commit/36024d1190db2726f847b16dddc011879b8260f3))
- Implement iterator over symetric difference between two `BitSet`s — ([a024487](https://github.com/GrayJack/rbitset/commit/a0244872ef0454126d40a9ddb8673ed8d2687c1d))
- Make `Debug` formatting for `Iter` and `IntoIter` similar to `HashSet`/`BtreeSet` — ([07a4f94](https://github.com/GrayJack/rbitset/commit/07a4f941d43bbc26e692a40b0db24f7e85d1b632))
- (**BREAKING**) Changed the debug printing of `BitSet`. — ([5045e2d](https://github.com/GrayJack/rbitset/commit/5045e2dfb5a29dd66461d44351c69bcb052776a5))
- (**BREAKING**) Rework again the debug implementation and move old implementation to `fmt::Binary` formatting — ([c4c0eb5](https://github.com/GrayJack/rbitset/commit/c4c0eb5ae84e58b7511d31137a4c3f1733633d99))
- (**BREAKING**) Change the insert to match the semantics of standard library `HashSet` and `BtreeSet` — ([afd79f9](https://github.com/GrayJack/rbitset/commit/afd79f97b664b3df5331bc9c1f767e7011de5a47))
- (**BREAKING**) Change the `remove` and `contains` to match the semantics of standard library HashSet and BtreeSet — ([a24c3cc](https://github.com/GrayJack/rbitset/commit/a24c3cc16175459a8814a14dd42c047f675f5c7b))
- Introduce a Drain iterator — ([478a33a](https://github.com/GrayJack/rbitset/commit/478a33a3b6f41b0107f58fd10c219c1b981ff042))

### 🚜 Refactor

- (**BREAKING**) Create a `IntoIter` type for `BitSet` — ([5d8ed79](https://github.com/GrayJack/rbitset/commit/5d8ed79e5e2943ca665352a9dc378fb1fe2e687b))

### 📚 Documentation

- Improve parts of the documentation — ([c5589af](https://github.com/GrayJack/rbitset/commit/c5589afd010895890b2c2e59df6b92d9d4649f96))
- Improve documentation for `Iter` related items — ([68e0e77](https://github.com/GrayJack/rbitset/commit/68e0e77f2dbf7e7c20ad9d2635b124e12c65d9f4))
- Improve documentation for `IntoIter` type — ([0b35df3](https://github.com/GrayJack/rbitset/commit/0b35df34eff2c32a683e9a96bbcdf6617960b796))
- Add examples to all functions part of the public API and reword a few items — ([8dc0383](https://github.com/GrayJack/rbitset/commit/8dc0383ac5be864aaccad11059a88d11021c2ea6))

## 0.1.0 — 2022-03-05

### 🚀 Features

- Initial implementation — ([1f33aaa](https://github.com/GrayJack/rbitset/commit/1f33aaa02c59dad2292e46e370300bc69b5699c4))
- Implement a fill() function — ([56dfd7f](https://github.com/GrayJack/rbitset/commit/56dfd7f649b76f65c86ab61770186db77a0e1f7b))
- Add safe ways to convert back and forth from underlying array — ([6807e88](https://github.com/GrayJack/rbitset/commit/6807e885017f982643025d6208ad97a8b995e8ce))
- Implement iterator for BitSet — ([84a4b20](https://github.com/GrayJack/rbitset/commit/84a4b20fed456bd1c2a277c21e9774eae218a33c))
- Implement Debug — ([ea610ef](https://github.com/GrayJack/rbitset/commit/ea610ef3824c15339474be40146550bcf584d469))
- Implement binary not — ([d2331de](https://github.com/GrayJack/rbitset/commit/d2331defe0ee8e45ebf2f56c0f47c8ba5cf3633d))

### 🚜 Refactor

- Apply clippy fixes — ([773eecd](https://github.com/GrayJack/rbitset/commit/773eecd40cfe8460ed4e7f7e8f495c8ea621122e))
- (**BREAKING**) Reimplement `BitSet` type based on const generics — ([c5fb1ff](https://github.com/GrayJack/rbitset/commit/c5fb1ff0a3985744e83731fa119d98c10d694086))

### ⚙️ Continuous Integration

- Configure Github Actions — ([9de18c1](https://github.com/GrayJack/rbitset/commit/9de18c10e8de307f33705cadc74560807b24f854))
- Fix ctrl+c ctrl+v problem — ([f5fec17](https://github.com/GrayJack/rbitset/commit/f5fec1767263364f45dce94419582222dc0d2d00))

<!-- generated by git-cliff -->

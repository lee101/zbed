`bed` is the filesystem-search CLI that ships with `zbed`.

It indexes:

- text files by filename and content lines
- binary/media files by filename only when `--search-binaries` is enabled

Examples:

```bash
./zig-out/bin/bed index . --search-binaries
./zig-out/bin/bed "database connection"
./zig-out/bin/bed "opus audio" --path /some/tree --search-binaries
```

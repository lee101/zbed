//! zbed – fast semantic code search powered by static embeddings.
//!
//! Pure Zig port of gobed (github.com/lee101/gobed).
//! Uses int8 quantized embeddings from sentence-transformers/static-retrieval-mrl-en-v1
//! with SIMD-accelerated cosine similarity search.

pub const tokenizer = @import("tokenizer.zig");
pub const embed = @import("embed.zig");
pub const search = @import("search.zig");
pub const index = @import("index.zig");

pub const Tokenizer = tokenizer.Tokenizer;
pub const EmbedModel = embed.EmbedModel;
pub const FlatIndex = search.FlatIndex;
pub const Index = index.Index;
pub const SearchResult = search.SearchResult;
pub const Document = index.Document;

pub const EMBED_DIM = embed.EMBED_DIM;
pub const MAX_EMBED_DIM = embed.MAX_EMBED_DIM;

// ─── tests ───────────────────────────────────────────────────────────
test {
    @import("std").testing.refAllDecls(@This());
}

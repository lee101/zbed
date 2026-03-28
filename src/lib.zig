//! zbed - quantized semantic filesystem search in Zig.

pub const tokenizer = @import("tokenizer.zig");
pub const embed = @import("embed.zig");
pub const search = @import("search.zig");
pub const index = @import("index.zig");
pub const server = @import("server.zig");

pub const Tokenizer = tokenizer.Tokenizer;
pub const EmbedModel = embed.EmbedModel;
pub const EmbedScratch = embed.EmbedScratch;
pub const QuantizedEmbedding = embed.QuantizedEmbedding;

pub const FlatIndex = search.FlatIndex;
pub const QuantizedFlatIndex = search.QuantizedFlatIndex;
pub const SearchResult = search.SearchResult;

pub const Index = index.Index;
pub const Document = index.Document;
pub const DocumentKind = index.DocumentKind;
pub const WalkOptions = index.WalkOptions;
pub const ServerState = server.ServerState;

pub const EMBED_DIM = embed.EMBED_DIM;
pub const MAX_EMBED_DIM = embed.MAX_EMBED_DIM;
pub const MAX_TOKENS = embed.MAX_TOKENS;

test {
    @import("std").testing.refAllDecls(@This());
}

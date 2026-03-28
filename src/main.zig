const cli = @import("cli.zig");

pub fn main() !void {
    try cli.runCli("zbed");
}

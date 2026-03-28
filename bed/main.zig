const cli = @import("zbed_cli");

pub fn main() !void {
    try cli.runCli("bed");
}

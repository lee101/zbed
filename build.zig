const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // --- library module (shared by exe + tests) ---
    const lib_mod = b.addModule("zbed", .{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    // --- executable ---
    const exe = b.addExecutable(.{
        .name = "zbed",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("zbed", lib_mod);
    b.installArtifact(exe);

    // --- run step ---
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run zbed");
    run_step.dependOn(&run_cmd.step);

    // --- unit tests ---
    const test_sources = [_][]const u8{
        "src/tokenizer.zig",
        "src/embed.zig",
        "src/search.zig",
        "src/index.zig",
        "src/lib.zig",
    };

    const test_step = b.step("test", "Run unit tests");

    for (test_sources) |src| {
        const t = b.addTest(.{
            .root_source_file = b.path(src),
            .target = target,
            .optimize = optimize,
        });
        const run_t = b.addRunArtifact(t);
        test_step.dependOn(&run_t.step);
    }
}

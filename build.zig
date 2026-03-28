const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib_mod = b.createModule(.{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    const cli_mod = b.createModule(.{
        .root_source_file = b.path("src/cli.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zbed", .module = lib_mod },
        },
    });

    const exe = b.addExecutable(.{
        .name = "zbed",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zbed", .module = lib_mod },
                .{ .name = "zbed_cli", .module = cli_mod },
            },
        }),
    });
    b.installArtifact(exe);

    const bed_exe = b.addExecutable(.{
        .name = "bed",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bed/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zbed", .module = lib_mod },
                .{ .name = "zbed_cli", .module = cli_mod },
            },
        }),
    });
    b.installArtifact(bed_exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run zbed");
    run_step.dependOn(&run_cmd.step);

    const test_sources = [_][]const u8{
        "src/tokenizer.zig",
        "src/embed.zig",
        "src/search.zig",
        "src/index.zig",
        "src/lib.zig",
        "src/cli.zig",
        "src/main.zig",
    };

    const test_step = b.step("test", "Run unit tests");

    for (test_sources) |src| {
        const t = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path(src),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "zbed", .module = lib_mod },
                    .{ .name = "zbed_cli", .module = cli_mod },
                },
            }),
        });
        const run_t = b.addRunArtifact(t);
        test_step.dependOn(&run_t.step);
    }
}

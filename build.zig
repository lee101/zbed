const std = @import("std");

fn linkCuda(step: *std.Build.Step.Compile, cuda_obj: std.Build.LazyPath, cuda_lib: []const u8) void {
    step.root_module.addObjectFile(cuda_obj);
    step.root_module.addLibraryPath(.{ .cwd_relative = cuda_lib });
    step.root_module.linkSystemLibrary("cudart", .{});
    step.linkLibC();
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const enable_cuda = b.option(bool, "cuda", "Enable CUDA backend") orelse false;
    const cuda_path = b.option([]const u8, "cuda-path", "Path to CUDA toolkit") orelse "/usr/local/cuda-12";
    const cuda_include = b.fmt("{s}/include", .{cuda_path});
    const cuda_lib = b.fmt("{s}/lib64", .{cuda_path});

    const build_opts = b.addOptions();
    build_opts.addOption(bool, "have_cuda", enable_cuda);

    const lib_mod = b.createModule(.{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib_mod.addOptions("build_options", build_opts);
    lib_mod.addIncludePath(b.path("src/cuda"));

    const cli_mod = b.createModule(.{
        .root_source_file = b.path("src/cli.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zbed", .module = lib_mod },
        },
    });
    cli_mod.addOptions("build_options", build_opts);
    cli_mod.addIncludePath(b.path("src/cuda"));

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

    if (enable_cuda) {
        const nvcc = b.addSystemCommand(&.{ "nvcc", "-O3", "-std=c++17", "-Xcompiler", "-fPIC", "-c", "src/cuda/zbed_cuda.cu" });
        nvcc.addArg("-o");
        const cuda_obj = nvcc.addOutputFileArg("zbed_cuda.o");
        nvcc.addArgs(&.{ "-I", "src/cuda", "-I" });
        nvcc.addArg(cuda_include);

        linkCuda(exe, cuda_obj, cuda_lib);
        linkCuda(bed_exe, cuda_obj, cuda_lib);
    }

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
        t.root_module.addOptions("build_options", build_opts);
        t.root_module.addIncludePath(b.path("src/cuda"));
        if (enable_cuda) {
            const nvcc = b.addSystemCommand(&.{ "nvcc", "-O3", "-std=c++17", "-Xcompiler", "-fPIC", "-c", "src/cuda/zbed_cuda.cu" });
            nvcc.addArg("-o");
            const cuda_obj = nvcc.addOutputFileArg("zbed_cuda.o");
            nvcc.addArgs(&.{ "-I", "src/cuda", "-I" });
            nvcc.addArg(cuda_include);
            linkCuda(t, cuda_obj, cuda_lib);
        }
        const run_t = b.addRunArtifact(t);
        test_step.dependOn(&run_t.step);
    }
}

const std = @import("std");
const testing = std.testing;
const BuiltinType = std.builtin.Type;
const Allocator = std.mem.Allocator;
const util = @import("util.zig");
const commons = @import("commons.zig");
const Shape = commons.Shape;
const Names = commons.Names;

pub const TensorError = error{ InvalidShape, InvalidNames };

pub fn Tensor(comptime T: type) type {
    return struct {
        slice: []const T,
        shape: Shape,
        names: Names,
        allocator: Allocator,
        const Self = @This();

        pub const TensorOpts = struct {
            shape: Shape,
            names: Names = &.{},
        };
        pub fn tensor(allocator: Allocator, slice: []const T, options: TensorOpts) !Self {
            if (slice.len != shapeSize(options.shape)) {
                return TensorError.InvalidShape;
            }
            if (options.names.len > 0 and options.names.len != options.shape.len) {
                return TensorError.InvalidNames;
            }
            return Self{
                .allocator = allocator,
                .slice = try allocator.dupe(T, slice),
                .shape = try allocator.dupe(usize, options.shape),
                .names = options.names,
            };
        }

        pub const IotaOpts = struct { axis: ?usize = null, names: Names = &.{} };
        pub fn iota(allocator: Allocator, shape: Shape, opts: IotaOpts) !Self {
            const shape_size = shapeSize(shape);
            var slice = try allocator.alloc(T, shape_size);
            for (0..shape_size) |i| {
                const shape_indexes = try idxToShapeIndexes(allocator, i, shape);
                defer allocator.free(shape_indexes);
                const val = if (opts.axis == null) i else shape_indexes[opts.axis.?];
                defer slice[i] = switch (@typeInfo(T)) {
                    .Int => @as(T, @intCast(val)),
                    .Float => @as(T, @floatFromInt(val)),
                    else => @compileError("float and int types accepted"),
                };
            }
            return Self{
                .allocator = allocator,
                .slice = slice,
                .shape = try allocator.dupe(usize, shape),
                .names = opts.names,
            };
        }

        pub const TriOpts = struct { k: usize = 0 };
        pub fn tri(allocator: Allocator, m: usize, n: usize, opts: TriOpts) !Self {
            var i: usize = 0;
            var j: usize = 0;
            var array = try std.ArrayList(u8).initCapacity(allocator, m * n);
            while (i < m) : (i += 1) {
                while (j < n) : (j += 1) {
                    try array.append(if (j <= i + opts.k) 1 else 0);
                }
                j = 0;
            }
            return Self{
                .allocator = allocator,
                .slice = try array.toOwnedSlice(),
                .shape = try allocator.dupe(usize, &.{ m, n }),
                .names = &.{},
            };
        }

        pub fn tril(self: *const Self, opts: TriOpts) !Self {
            const shape_size = shapeSize(self.shape);
            var res = try self.allocator.alloc(T, shape_size);
            for (self.slice, 0..) |value, i| {
                const idxs = try idxToShapeIndexes(self.allocator, i, self.shape);
                defer self.allocator.free(idxs);
                const j = idxs[idxs.len - 2];
                const k = idxs[idxs.len - 1];
                res[i] = if (k <= j + opts.k) value else 0;
            }
            return Self{
                .allocator = self.allocator,
                .slice = res,
                .shape = self.shape,
                .names = self.names,
            };
        }

        pub fn triu(self: *const Self, opts: TriOpts) !Self {
            const shape_size = shapeSize(self.shape);
            var res = try self.allocator.alloc(T, shape_size);
            for (self.slice, 0..) |value, i| {
                const idxs = try idxToShapeIndexes(self.allocator, i, self.shape);
                defer self.allocator.free(idxs);
                const j = idxs[idxs.len - 2];
                const k = idxs[idxs.len - 1];
                res[i] = if (k >= j + opts.k) value else 0;
            }
            return Self{
                .allocator = self.allocator,
                .slice = res,
                .shape = self.shape,
                .names = self.names,
            };
        }

        const DiagOpts = struct { offset: i32 = 0 };
        pub fn makeDiagonal(self: *Self, opts: DiagOpts) !Self {
            if (self.shape.len != 1) {
                @panic("shape.len > 1, Todo while doing vectorized tensors");
            }
            const side_count = self.shape[0] + @as(usize, @intCast(util.abs(opts.offset)));
            const shape: Shape = &.{ side_count, side_count };
            var slice = try util.slice_init(T, self.allocator, side_count * side_count, 0);

            for (self.slice, 0..) |value, i| {
                const idx = shapeIndexesToIdx(&.{ i, i + @as(usize, @intCast(opts.offset)) }, shape);
                slice[idx] = value;
            }
            return Self{
                .allocator = self.allocator,
                .slice = slice,
                .shape = try self.allocator.dupe(usize, shape),
                .names = &.{},
            };
        }

        // Todo : tensor version
        pub const LinspaceOpts = struct { endpoint: bool = true, name: ?[]const u8 = null };
        pub fn linspace(allocator: Allocator, start: usize, stop: usize, n: usize, opts: LinspaceOpts) !Self {
            var slice = try allocator.alloc(T, n);
            const minus: usize = if (opts.endpoint) 1 else 0;
            var step: f64 = (@as(f64, @floatFromInt(stop)) - @as(f64, @floatFromInt(start))) / @as(f64, @floatFromInt(n - minus));
            var val: f64 = @as(f64, @floatFromInt(start));
            for (0..n) |i| {
                slice[i] = switch (@typeInfo(T)) {
                    .Int => @as(T, @intFromFloat(val)),
                    .Float => val,
                    else => @compileError("float and int types accepted"),
                };
                val += step;
            }
            return Self{
                .allocator = allocator,
                .slice = slice,
                .names = if (opts.name == null) &.{} else &.{opts.name.?},
                .shape = try allocator.dupe(usize, &.{n}),
            };
        }

        pub const AggregatesOtps = struct { axes: []const usize = &.{}, keep_axes: bool = false };
        pub fn product(self: *Self, opts: AggregatesOtps) !Self {
            if (opts.axes.len == 0) {
                var res: T = 1;
                for (self.slice) |it| {
                    if (it == 0) {
                        res = 0;
                        break;
                    }
                    res *= it;
                }
                return Self{ .allocator = self.allocator, .slice = try self.allocator.alloc(T, 1), .names = &.{}, .shape = try self.allocator.dupe(usize, &.{}) };
            }

            var result_shape = try util.shapeRemoveAxis(self.allocator, self.shape, opts.axes);
            var result_shape_size = shapeSize(result_shape);
            var slice = try util.slice_init(T, self.allocator, result_shape_size, 1);

            for (self.slice, 0..) |s, i| {
                var original_indices = try idxToShapeIndexes(self.allocator, i, self.shape);
                defer self.allocator.free(original_indices);
                var destination_indexes = try util.shapeRemoveAxis(self.allocator, original_indices, opts.axes);
                defer self.allocator.free(destination_indexes);
                var destination_idx = shapeIndexesToIdx(destination_indexes, result_shape);
                slice[destination_idx] *= s;
            }

            var final_shape = if (opts.keep_axes)
                try util.shapeKeepAxes(self.allocator, self.shape, opts.axes)
            else
                result_shape;

            return Self{ .allocator = self.allocator, .slice = slice, .names = &.{}, .shape = final_shape };
        }

        pub fn sum(self: *Self, opts: AggregatesOtps) !Self {
            if (opts.axes.len == 0) {
                var res: T = 1;
                for (self.slice) |it| {
                    res += it;
                }
                return Self{ .allocator = self.allocator, .slice = try self.allocator.alloc(T, 1), .names = &.{}, .shape = try self.allocator.dupe(usize, &.{}) };
            }

            var result_shape = try util.shapeRemoveAxis(self.allocator, self.shape, opts.axes);
            var result_shape_size = shapeSize(result_shape);
            var slice = try util.slice_init(T, self.allocator, result_shape_size, 0);

            for (self.slice, 0..) |s, i| {
                var original_indices = try idxToShapeIndexes(self.allocator, i, self.shape);
                defer self.allocator.free(original_indices);
                var destination_indexes = try util.shapeRemoveAxis(self.allocator, original_indices, opts.axes);
                defer self.allocator.free(destination_indexes);
                var destination_idx = shapeIndexesToIdx(destination_indexes, result_shape);
                slice[destination_idx] += s;
            }

            var final_shape = if (opts.keep_axes)
                try util.shapeKeepAxes(self.allocator, self.shape, opts.axes)
            else
                result_shape;

            return Self{ .allocator = self.allocator, .slice = slice, .names = &.{}, .shape = final_shape };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.slice);
            self.slice = undefined;
            self.allocator.free(self.shape);
            self.shape = undefined;
        }

        pub fn toString(self: Self) ![]u8 {
            const sliceChildType = childType(self.slice);
            const shapeString = try shapeToString(self.allocator, self.shape, self.names);
            defer self.allocator.free(shapeString);
            var list = std.ArrayList(u8).init(self.allocator);

            try list.writer().print(
                "#nz.Tensor<\n {s}{s} \n {any} \n>",
                .{ sliceChildType, shapeString, self.slice },
            );
            return list.toOwnedSlice();
        }

        pub fn debug_print(self: *Self) !void {
            const str = try self.toString();
            defer self.allocator.free(str);
            std.debug.print("{s}\n", .{str});
        }
    };
}

fn childType(slice: anytype) []const u8 {
    const type_info = @typeInfo(@TypeOf(slice));
    switch (type_info) {
        .Pointer => |info| if (info.size == .Slice) {
            return @typeName(info.child);
        },
        else => return "",
    }
}

pub fn shapeSize(shape: Shape) usize {
    var res: usize = 1;
    for (shape) |dim_size| {
        res *= dim_size;
    }
    return res;
}

pub fn idxToShapeIndexes(allocator: Allocator, idx: usize, shape: Shape) !Shape {
    var slice = try allocator.alloc(usize, shape.len);
    var shape_size = shapeSize(shape);
    var tempIdx = idx;
    for (shape, 0..) |s, i| {
        const ratio: usize = try std.math.divFloor(usize, tempIdx, shape_size / s);
        tempIdx -= (ratio * (shape_size / s));
        shape_size /= s;
        slice[i] = ratio;
    }
    return slice;
}

pub fn shapeIndexesToIdx(idxs: Shape, shape: Shape) usize {
    var shape_size = shapeSize(shape);
    var res: usize = 0;
    for (shape, idxs) |s, i| {
        const elem_count = shape_size / s;
        res += i * elem_count;
        shape_size = elem_count;
    }
    return res;
}

fn shapeToString(allocator: Allocator, shape: Shape, names: Names) ![]u8 {
    var result = std.ArrayList(u8).init(allocator);
    if (names.len == 0) {
        for (shape) |it| {
            try result.writer().print("[{}]", .{it});
        }
        return result.toOwnedSlice();
    }
    for (names, shape) |n, s| {
        if (n != null) {
            try result.writer().print("[{s}: {}]", .{ n.?, s });
        } else {
            try result.writer().print("[{}]", .{s});
        }
    }
    return result.toOwnedSlice();
}

test "some tests" {
    const print = std.debug.print;
    _ = print;
    const allocator = testing.allocator;
    // const TensorF64 = Tensor(f64);
    // var tensor = try TensorF64.tensor(allocator, &[_]f64{ 1, 2, 3, 4, 5, 5 }, .{ .shape = &.{ 2, 3 }, .names = &[2]?[]const u8{ "x", "y" } });
    // defer tensor.deinit();
    // var a_tri = try Tensor(u8).tri(allocator, 4, 5, .{ .k = 2 });
    // defer a_tri.deinit();
    // const tri_string = try a_tri.toString();
    // defer allocator.free(tri_string);
    // var a_tril = try Tensor(f64).tril(&tensor, .{});
    // defer a_tril.deinit();
    // const tril_string = try a_tril.toString();
    // defer allocator.free(tril_string);
    // print("{s}", .{tril_string});
    // var a_triu = try Tensor(f64).triu(&tensor, .{ .k = 0 });
    // defer a_triu.deinit();
    // const triu_string = try a_triu.toString();
    // defer allocator.free(triu_string);
    // print("{s}\n", .{triu_string});
    // const indexes = try idxToShapeIndexes(allocator, 52, &.{ 2, 3, 3, 4 });
    // defer allocator.free(indexes);
    // // std.debug.print("{any}", .{indexes});
    // const index = shapeIndexesToIdx(&.{ 1, 1, 1 }, &.{ 2, 3, 3 });
    // try testing.expectEqual(index, 13);
    // var one_dim_tensor = try TensorF64.tensor(allocator, &[4]f64{ 1, 2, 3, 4 }, .{ .shape = &.{4} });
    // defer one_dim_tensor.deinit();
    // var diag = try TensorF64.makeDiagonal(&one_dim_tensor, .{ .offset = 1 });
    // defer diag.deinit();
    // const diag_string = try diag.toString();
    // defer allocator.free(diag_string);
    // print("diag: {s}\n", .{diag_string});
    // var iot = try Tensor(f64).iota(allocator, &.{ 2, 3, 4 }, .{ .axis = 1 });
    // defer iot.deinit();
    // const iot_string = try iot.toString();
    // defer allocator.free(iot_string);
    // print("iot_string: {s}\n", .{iot_string});
    // var lint = try Tensor(f64).linspace(allocator, 0, 10, 5, .{ .endpoint = true, .name = "x" });
    // defer lint.deinit();
    // const lint_string = try lint.toString();
    // defer allocator.free(lint_string);
    // print("linespace: {s}\n", .{lint_string});
    //
    // var iotap = try Tensor(u32).iota(allocator, &.{ 2, 2, 3 }, .{ .names = &.{ "x", "y", "z" } });
    // defer iotap.deinit();
    // try iotap.debug_print();
    // var prod = try iotap.product(.{});
    // defer prod.deinit();
    // try prod.debug_print();

    var iotaq = try Tensor(u32).iota(allocator, &.{ 2, 2, 3 }, .{ .names = &.{ "x", "y", "z" } });
    defer iotaq.deinit();
    try iotaq.debug_print();
    var prodq = try iotaq.product(.{ .axes = &.{0} });
    defer prodq.deinit();
    try prodq.debug_print();
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    var iotaq = try Tensor(u32).iota(allocator, &.{ 2, 2, 3 }, .{ .names = &.{ "x", "y", "z" } });
    defer iotaq.deinit();
    try iotaq.debug_print();
    var prodq = try iotaq.product(.{ .keep_axes = true, .axes = &.{ 1, 2 } });
    defer prodq.deinit();
    try prodq.debug_print();

    var sumq = try iotaq.sum(.{ .axes = &.{ 1, 2 } });
    defer sumq.deinit();
    try sumq.debug_print();
}

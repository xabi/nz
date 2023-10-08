const std = @import("std");
const testing = std.testing;
const BuiltinType = std.builtin.Type;
const Allocator = std.mem.Allocator;
const util = @import("util.zig");

export fn add(a: i32, b: i32) i32 {
    return a + b;
}

test "basic add functionality" {
    try testing.expect(add(3, 7) == 10);
}

pub const TensorError = error{ InvalidShape, InvalidNames };

const Shape = []const usize;
const Names = ?[]const ?[]const u8;

pub fn Tensor(comptime T: type) type {
    return struct {
        slice: []const T,
        shape: Shape,
        names: Names,
        allocator: Allocator,
        const Self = @This();

        pub const TensorOpts = struct {
            shape: Shape,
            names: Names = null,
        };
        pub fn tensor(allocator: Allocator, slice: []const T, options: TensorOpts) !Self {
            if (slice.len != shapeSize(options.shape)) {
                return TensorError.InvalidShape;
            }
            if (options.names != null and options.names.?.len > 0 and options.names.?.len != options.shape.len) {
                return TensorError.InvalidNames;
            }
            return Self{
                .allocator = allocator,
                .slice = try allocator.dupe(T, slice),
                .shape = options.shape,
                .names = options.names,
            };
        }

        pub fn iota(allocator: Allocator, shape: Shape) !Self {
            const shape_size = shapeSize(shape);
            var slice = try allocator.alloc(T, shape_size);
            for (0..shape_size) |i| {
                slice[i] = i;
            }
            return Self{
                .allocator = allocator,
                .slice = slice,
                .shape = shape,
                .names = null,
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
                .shape = &.{ m, n },
                .names = null,
            };
        }

        pub fn tril(self: *const Self, opts: TriOpts) !Self {
            const shape_size = shapeSize(self.shape);
            var res = try self.allocator.alloc(T, shape_size);
            for (self.slice, 0..) |value, i| {
                const idxs = try Self.idxToShapeIndexes(self.allocator, i, self.shape);
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
                const idxs = try Self.idxToShapeIndexes(self.allocator, i, self.shape);
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
            var slice = try self.allocator.alloc(T, side_count * side_count);
            for (slice) |*v| {
                v.* = 0;
            }
            for (self.slice, 0..) |value, i| {
                const idx = Self.shapeIndexesToIdx(&.{ i, i + @as(usize, @intCast(opts.offset)) }, shape);
                slice[idx] = value;
            }
            return Self{
                .allocator = self.allocator,
                .slice = slice,
                .shape = shape,
                .names = null,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.slice);
            self.slice = undefined;
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

fn shapeToString(allocator: Allocator, shape: Shape, names: Names) ![]u8 {
    var result = std.ArrayList(u8).init(allocator);
    if (names == null) {
        for (shape) |it| {
            try result.writer().print("[{}]", .{it});
        }
        return result.toOwnedSlice();
    }
    for (names.?, shape) |n, s| {
        if (n != null) {
            try result.writer().print("[{s}: {}]", .{ n.?, s });
        } else {
            try result.writer().print("[{}]", .{s});
        }
    }
    return result.toOwnedSlice();
}

test "some tests" {
    const allocator = testing.allocator;
    const TensorF64 = Tensor(f64);
    var tensor = try TensorF64.tensor(allocator, &[_]f64{ 1, 2, 3, 4, 5, 5 }, .{ .shape = &.{ 2, 3 }, .names = &[2]?[]const u8{ "x", "y" } });
    defer tensor.deinit();
    var a_tri = try Tensor(u8).tri(allocator, 4, 5, .{ .k = 2 });
    defer a_tri.deinit();
    const tri_string = try a_tri.toString();
    defer allocator.free(tri_string);
    var a_tril = try Tensor(f64).tril(&tensor, .{});
    defer a_tril.deinit();
    const tril_string = try a_tril.toString();
    defer allocator.free(tril_string);
    std.debug.print("{s}", .{tril_string});
    var a_triu = try Tensor(f64).triu(&tensor, .{ .k = 0 });
    defer a_triu.deinit();
    const triu_string = try a_triu.toString();
    defer allocator.free(triu_string);
    std.debug.print("{s}\n", .{triu_string});
    const indexes = try TensorF64.idxToShapeIndexes(allocator, 52, &.{ 2, 3, 3, 4 });
    defer allocator.free(indexes);
    // std.debug.print("{any}", .{indexes});
    const index = TensorF64.shapeIndexesToIdx(&.{ 1, 1, 1 }, &.{ 2, 3, 3 });
    try testing.expectEqual(index, 13);
    var one_dim_tensor = try TensorF64.tensor(allocator, &[4]f64{ 1, 2, 3, 4 }, .{ .shape = &.{4} });
    defer one_dim_tensor.deinit();
    var diag = try TensorF64.makeDiagonal(&one_dim_tensor, .{ .offset = 1 });
    defer diag.deinit();
    const diag_string = try diag.toString();
    defer allocator.free(diag_string);
    std.debug.print("diag: {s}\n", .{diag_string});
    var iot = try Tensor(usize).iota(allocator, &.{ 1, 2, 3 });
    defer iot.deinit();
    const iot_string = try iot.toString();
    defer allocator.free(iot_string);
    std.debug.print("iot_string: {s}", .{iot_string});
}

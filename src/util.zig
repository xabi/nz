const std = @import("std");
const commons = @import("commons.zig");
const Shape = commons.Shape;
const Names = commons.Names;
const Allocator = std.mem.Allocator;
const testing = std.testing;
const print = std.debug.print;

pub fn abs(num: i32) i32 {
    if (num < 0) return -num;
    return num;
}

pub fn shapeRemoveAxis(allocator: Allocator, shape: Shape, axis: []const usize) !Shape {
    var result_shape = try allocator.alloc(usize, shape.len - axis.len);
    var j: usize = 0;
    var a: usize = 0;
    for (shape, 0..) |s, i| {
        while (a < axis.len) : (a += 1) {
            if (axis[a] == i) {
                break;
            }
        } else {
            result_shape[j] = s;
            j += 1;
        }
        a = 0;
    }
    return result_shape;
}

pub fn shapeKeepAxes(allocator: Allocator, shape: Shape, axes: []const usize) !Shape {
    var result_shape = try allocator.dupe(usize, shape);
    for (axes) |a| {
        result_shape[a] = 1;
    }
    return result_shape;
}

pub fn slice_init(comptime T: type, allocator: Allocator, size: usize, value: T) ![]T {
    var slice = try allocator.alloc(T, size);
    for (slice) |*s| {
        s.* = value;
    }
    return slice;
}

pub fn shapeSize(shape: Shape) usize {
    var res: usize = 1;
    for (shape) |dim_size| {
        res *= dim_size;
    }
    return res;
}

pub fn min_value(comptime T: type) T {
    return switch (@typeInfo(T)) {
        .Int => std.math.minInt(T),
        .Float => std.math.floatMin(T),
        else => @compileError("float and int types accepted"),
    };
}

pub fn max_value(comptime T: type) T {
    return switch (@typeInfo(T)) {
        .Int => std.math.maxInt(T),
        .Float => std.math.floatMax(T),
        else => @compileError("float and int types accepted"),
    };
}

pub fn remaingShapeSize(shape: Shape, axes: []const usize) usize {
    var res: usize = 1;
    for (axes) |a| {
        res *= shape[a];
    }
    return res;
}

pub fn shapeToString(allocator: Allocator, shape: Shape, names: Names) ![]u8 {
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

pub fn childType(slice: anytype) []const u8 {
    const type_info = @typeInfo(@TypeOf(slice));
    switch (type_info) {
        .Pointer => |info| if (info.size == .Slice) {
            return @typeName(info.child);
        },
        else => return "",
    }
}
test "util tests" {
    const shape: Shape = &.{ 2, 2, 3 };
    const axis: []const usize = &.{ 0, 2 };
    const remaining = try shapeRemoveAxis(testing.allocator, shape, axis);
    defer testing.allocator.free(remaining);
    try testing.expectEqualSlices(usize, &.{2}, remaining);
    const axis1: []const usize = &.{ 0, 1 };
    const remaining1 = try shapeRemoveAxis(testing.allocator, shape, axis1);
    defer testing.allocator.free(remaining1);
    try testing.expectEqualSlices(usize, &.{3}, remaining1);

    const axis2: []const usize = &.{0};
    const remaining2 = try shapeRemoveAxis(testing.allocator, shape, axis2);
    defer testing.allocator.free(remaining2);
    try testing.expectEqualSlices(usize, &.{ 2, 3 }, remaining2);
}

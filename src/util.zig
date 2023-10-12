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

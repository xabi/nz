const std = @import("std");
const testing = std.testing;
const BuiltinType = std.builtin.Type;
const Allocator = std.mem.Allocator;
const util = @import("util.zig");
const commons = @import("commons.zig");
const Op = @import("op.zig").Op;
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
            if (slice.len != util.shapeSize(options.shape)) {
                return TensorError.InvalidShape;
            }
            if (options.names.len > 0 and options.names.len != options.shape.len) {
                return TensorError.InvalidNames;
            }
            return Self{
                .allocator = allocator,
                .slice = try allocator.dupe(T, slice),
                .shape = try allocator.dupe(usize, options.shape),
                .names = try allocator.dupe(?[]const u8, options.names),
            };
        }

        pub const IotaOpts = struct { axis: ?usize = null, names: Names = &.{} };
        pub fn iota(allocator: Allocator, shape: Shape, opts: IotaOpts) !Self {
            const shape_size = util.shapeSize(shape);
            var slice = try allocator.alloc(T, shape_size);
            for (0..shape_size) |i| {
                const shape_indexes = try util.idxToShapeIndexes(allocator, i, shape);
                defer allocator.free(shape_indexes);
                const val = if (opts.axis == null) i else shape_indexes[opts.axis.?];
                slice[i] = switch (@typeInfo(T)) {
                    .Int => @as(T, @intCast(val)),
                    .Float => @as(T, @floatFromInt(val)),
                    else => @compileError("float and int types accepted"),
                };
            }
            return Self{
                .allocator = allocator,
                .slice = slice,
                .shape = try allocator.dupe(usize, shape),
                .names = try allocator.dupe(?[]const u8, opts.names),
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
            const shape_size = util.shapeSize(self.shape);
            var res = try self.allocator.alloc(T, shape_size);
            for (self.slice, 0..) |value, i| {
                const idxs = try util.idxToShapeIndexes(self.allocator, i, self.shape);
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
            const shape_size = util.shapeSize(self.shape);
            var res = try self.allocator.alloc(T, shape_size);
            for (self.slice, 0..) |value, i| {
                const idxs = try util.idxToShapeIndexes(self.allocator, i, self.shape);
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
                const idx = util.shapeIndexesToIdx(&.{ i, i + @as(usize, @intCast(opts.offset)) }, shape);
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
                return Self{ .allocator = self.allocator, .slice = try self.allocator.dupe(T, &.{res}), .names = &.{}, .shape = try self.allocator.dupe(usize, &.{}) };
            }

            var result_shape = try util.shapeRemoveAxis(self.allocator, self.shape, opts.axes);
            var result_shape_size = util.shapeSize(result_shape);
            var slice = try util.slice_init(T, self.allocator, result_shape_size, 1);

            for (self.slice, 0..) |s, i| {
                var original_indices = try util.idxToShapeIndexes(self.allocator, i, self.shape);
                defer self.allocator.free(original_indices);
                var destination_indexes = try util.shapeRemoveAxis(self.allocator, original_indices, opts.axes);
                defer self.allocator.free(destination_indexes);
                var destination_idx = util.shapeIndexesToIdx(destination_indexes, result_shape);
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
                var res: T = 0;
                for (self.slice) |it| {
                    res += it;
                }
                return Self{ .allocator = self.allocator, .slice = try self.allocator.dupe(T, &.{res}), .names = &.{}, .shape = try self.allocator.dupe(usize, &.{}) };
            }

            var result_shape = try util.shapeRemoveAxis(self.allocator, self.shape, opts.axes);
            var result_shape_size = util.shapeSize(result_shape);
            var slice = try util.slice_init(T, self.allocator, result_shape_size, 0);

            for (self.slice, 0..) |s, i| {
                var original_indices = try util.idxToShapeIndexes(self.allocator, i, self.shape);
                defer self.allocator.free(original_indices);
                var destination_indexes = try util.shapeRemoveAxis(self.allocator, original_indices, opts.axes);
                defer self.allocator.free(destination_indexes);
                var destination_idx = util.shapeIndexesToIdx(destination_indexes, result_shape);
                slice[destination_idx] += s;
            }

            var final_shape = if (opts.keep_axes)
                try util.shapeKeepAxes(self.allocator, self.shape, opts.axes)
            else
                result_shape;

            return Self{ .allocator = self.allocator, .slice = slice, .names = &.{}, .shape = final_shape };
        }

        pub fn all(self: *Self, opts: AggregatesOtps) !Tensor(u8) {
            if (opts.axes.len == 0) {
                var res: u8 = 1;
                for (self.slice) |it| {
                    if (it == 0) {
                        res = 0;
                        break;
                    }
                }
                return Tensor(u8){ .allocator = self.allocator, .slice = try self.allocator.dupe(u8, &.{res}), .names = &.{}, .shape = try self.allocator.dupe(usize, &.{}) };
            }

            var result_shape = try util.shapeRemoveAxis(self.allocator, self.shape, opts.axes);
            var result_shape_size = util.shapeSize(result_shape);
            var slice = try util.slice_init(u8, self.allocator, result_shape_size, 1);

            for (self.slice, 0..) |s, i| {
                var original_indices = try util.idxToShapeIndexes(self.allocator, i, self.shape);
                defer self.allocator.free(original_indices);
                var destination_indexes = try util.shapeRemoveAxis(self.allocator, original_indices, opts.axes);
                defer self.allocator.free(destination_indexes);
                var destination_idx = util.shapeIndexesToIdx(destination_indexes, result_shape);
                if (s == 0) {
                    slice[destination_idx] = 0;
                }
            }

            var final_shape = if (opts.keep_axes)
                try util.shapeKeepAxes(self.allocator, self.shape, opts.axes)
            else
                result_shape;

            return Tensor(u8){ .allocator = self.allocator, .slice = slice, .names = &.{}, .shape = final_shape };
        }

        pub fn any(self: *Self, opts: AggregatesOtps) !Tensor(u8) {
            if (opts.axes.len == 0) {
                var res: u8 = 0;
                for (self.slice) |it| {
                    if (it != 0) {
                        res = 1;
                        break;
                    }
                }
                return Tensor(u8){ .allocator = self.allocator, .slice = try self.allocator.dupe(u8, &.{res}), .names = &.{}, .shape = try self.allocator.dupe(usize, &.{}) };
            }

            var result_shape = try util.shapeRemoveAxis(self.allocator, self.shape, opts.axes);
            var result_shape_size = util.shapeSize(result_shape);
            var slice = try util.slice_init(u8, self.allocator, result_shape_size, 0);

            for (self.slice, 0..) |s, i| {
                var original_indices = try util.idxToShapeIndexes(self.allocator, i, self.shape);
                defer self.allocator.free(original_indices);
                var destination_indexes = try util.shapeRemoveAxis(self.allocator, original_indices, opts.axes);
                defer self.allocator.free(destination_indexes);
                var destination_idx = util.shapeIndexesToIdx(destination_indexes, result_shape);
                if (s != 0) {
                    slice[destination_idx] = 1;
                }
            }

            var final_shape = if (opts.keep_axes)
                try util.shapeKeepAxes(self.allocator, self.shape, opts.axes)
            else
                result_shape;

            return Tensor(u8){ .allocator = self.allocator, .slice = slice, .names = &.{}, .shape = final_shape };
        }

        pub fn reduceMax(self: *Self, opts: AggregatesOtps) !Self {
            var result_shape = if (opts.axes.len > 0)
                try util.shapeRemoveAxis(self.allocator, self.shape, opts.axes)
            else
                try self.allocator.dupe(usize, &.{});
            var result_shape_size = util.shapeSize(result_shape);
            var slice = try util.slice_init(T, self.allocator, result_shape_size, util.min_value(T));

            for (self.slice, 0..) |s, i| {
                var destination_idx: usize = 0;
                if (opts.axes.len > 0) {
                    destination_idx = dest_blk: {
                        var original_indices = try util.idxToShapeIndexes(self.allocator, i, self.shape);
                        defer self.allocator.free(original_indices);
                        var destination_indexes = try util.shapeRemoveAxis(self.allocator, original_indices, opts.axes);
                        defer self.allocator.free(destination_indexes);
                        break :dest_blk util.shapeIndexesToIdx(destination_indexes, result_shape);
                    };
                }
                if (s > slice[destination_idx]) {
                    slice[destination_idx] = s;
                }
            }
            var final_shape = if (opts.keep_axes)
                try util.shapeKeepAxes(self.allocator, self.shape, opts.axes)
            else
                result_shape;

            return Self{ .allocator = self.allocator, .slice = slice, .names = &.{}, .shape = final_shape };
        }

        pub fn reduceMin(self: *Self, opts: AggregatesOtps) !Self {
            var result_shape = if (opts.axes.len > 0)
                try util.shapeRemoveAxis(self.allocator, self.shape, opts.axes)
            else
                try self.allocator.dupe(usize, &.{});
            var result_shape_size = util.shapeSize(result_shape);
            var slice = try util.slice_init(T, self.allocator, result_shape_size, util.max_value(T));

            for (self.slice, 0..) |s, i| {
                var destination_idx: usize = 0;
                if (opts.axes.len > 0) {
                    destination_idx = dest_blk: {
                        var original_indices = try util.idxToShapeIndexes(self.allocator, i, self.shape);
                        defer self.allocator.free(original_indices);
                        var destination_indexes = try util.shapeRemoveAxis(self.allocator, original_indices, opts.axes);
                        defer self.allocator.free(destination_indexes);
                        break :dest_blk util.shapeIndexesToIdx(destination_indexes, result_shape);
                    };
                }
                if (s < slice[destination_idx]) {
                    slice[destination_idx] = s;
                }
            }
            var final_shape = if (opts.keep_axes)
                try util.shapeKeepAxes(self.allocator, self.shape, opts.axes)
            else
                result_shape;

            return Self{ .allocator = self.allocator, .slice = slice, .names = &.{}, .shape = final_shape };
        }

        pub fn mean(self: *Self, opts: AggregatesOtps) !Tensor(f64) {
            var sum_tensor = try self.sum(opts);
            defer sum_tensor.deinit();

            const length = @as(f64, @floatFromInt(util.remaingShapeSize(self.shape, opts.axes)));

            var slice = try self.allocator.alloc(f64, sum_tensor.slice.len);
            for (sum_tensor.slice, 0..) |it, i| {
                const itf64 = switch (@typeInfo(T)) {
                    .Float => @as(f64, it),
                    .Int => @as(f64, @floatFromInt(it)),
                    else => @compileError("not an int or float type"),
                };
                slice[i] = itf64 / length;
            }

            return Tensor(f64){
                .allocator = self.allocator,
                .slice = slice,
                .shape = try self.allocator.dupe(usize, sum_tensor.shape),
                .names = try self.allocator.dupe(?[]const u8, sum_tensor.names),
            };
        }

        pub fn elementWiseOpI(self: *Self, op: Op) !Self {
            const type_info = @typeInfo(T);
            if (type_info == .Int) {
                return self.copy();
            } else {
                var slice = try self.allocator.alloc(T, self.slice.len);
                for (self.slice, 0..) |value, i| {
                    slice[i] = switch (op) {
                        .abs => if (value < 0) -value else value,
                        .round => std.math.round(value),
                        .ceil => std.math.ceil(value),
                        else => unreachable,
                    };
                }

                return Self{
                    .allocator = self.allocator,
                    .slice = slice,
                    .shape = try self.allocator.dupe(usize, self.shape),
                    .names = try self.allocator.dupe(?[]const u8, self.names),
                };
            }
        }

        pub fn abs(self: *Self) !Self {
            const type_info = @typeInfo(T);
            if (type_info == .Int and type_info.Int.signedness == .unsigned) {
                return self.copy();
            } else {
                return elementWiseOpI(self, .abs);
            }
        }
        pub fn round(self: *Self) !Self {
            return elementWiseOpI(self, .round);
        }
        pub fn ceil(self: *Self) !Self {
            return elementWiseOpI(self, .ceil);
        }

        // element wise operations returning float tensors
        pub fn elementWiseOpF(self: *Self, op: Op) !Tensor(f64) {
            var slice = try self.allocator.alloc(f64, self.slice.len);
            for (self.slice, 0..) |value, i| {
                const float_val = switch (@typeInfo(T)) {
                    .Int => @as(f64, @intFromFloat(value)),
                    .Float => value,
                    else => @compileError("only float and integer types are accepted"),
                };

                slice[i] = switch (op) {
                    .cos => std.math.cos(float_val),
                    .acos => std.math.acos(float_val),
                    .cosh => std.math.cosh(float_val),
                    .acosh => std.math.acosh(float_val),
                    .tan => std.math.tan(float_val),
                    .atan => std.math.atan(float_val),
                    .tanh => std.math.tanh(float_val),
                    .atanh => std.math.atanh(float_val),
                    .sin => std.math.sin(float_val),
                    .asin => std.math.asin(float_val),
                    .sinh => std.math.sinh(float_val),
                    .asinh => std.math.asinh(float_val),
                    else => unreachable,
                };
            }
            return Tensor(f64){
                .allocator = self.allocator,
                .slice = slice,
                .shape = try self.allocator.dupe(usize, self.shape),
                .names = try self.allocator.dupe(?[]const u8, self.names),
            };
        }

        pub fn cos(self: *Self) !Tensor(f64) {
            return elementWiseOpF(self, Op.cos);
        }

        pub fn acos(self: *Self) !Tensor(f64) {
            return elementWiseOpF(self, Op.acos);
        }

        pub fn cosh(self: *Self) !Tensor(f64) {
            return elementWiseOpF(self, Op.cosh);
        }

        pub fn acosh(self: *Self) !Tensor(f64) {
            return elementWiseOpF(self, Op.acosh);
        }

        pub fn sin(self: *Self) !Tensor(f64) {
            return elementWiseOpF(self, Op.sin);
        }

        pub fn asin(self: *Self) !Tensor(f64) {
            return elementWiseOpF(self, Op.asin);
        }

        pub fn sinh(self: *Self) !Tensor(f64) {
            return elementWiseOpF(self, Op.sinh);
        }

        pub fn asinh(self: *Self) !Tensor(f64) {
            return elementWiseOpF(self, Op.asinh);
        }

        pub fn tan(self: *Self) !Tensor(f64) {
            return elementWiseOpF(self, Op.tan);
        }

        pub fn atan(self: *Self) !Tensor(f64) {
            return elementWiseOpF(self, Op.atan);
        }

        pub fn tanh(self: *Self) !Tensor(f64) {
            return elementWiseOpF(self, Op.tanh);
        }

        pub fn atanh(self: *Self) !Tensor(f64) {
            return elementWiseOpF(self, Op.atanh);
        }

        pub fn elementWiseBinaryOp(self: *Self, other: *Self, op: Op) !Tensor(f64) {
            _ = other;
            var slice = try self.allocator.alloc(f64, self.slice.len);
            for (self.slice, 0..) |value, i| {
                const float_val = switch (@typeInfo(T)) {
                    .Int => @as(f64, @intFromFloat(value)),
                    .Float => value,
                    else => @compileError("only float and integer types are accepted"),
                };

                slice[i] = switch (op) {
                    .cos => std.math.cos(float_val),
                    .acos => std.math.acos(float_val),
                    .cosh => std.math.cosh(float_val),
                    .acosh => std.math.acosh(float_val),
                    .tan => std.math.tan(float_val),
                    .atan => std.math.atan(float_val),
                    .tanh => std.math.tanh(float_val),
                    .atanh => std.math.atanh(float_val),
                    .sin => std.math.sin(float_val),
                    .asin => std.math.asin(float_val),
                    .sinh => std.math.sinh(float_val),
                    .asinh => std.math.asinh(float_val),
                    else => unreachable,
                };
            }
            return Tensor(f64){
                .allocator = self.allocator,
                .slice = slice,
                .shape = try self.allocator.dupe(usize, self.shape),
                .names = try self.allocator.dupe(?[]const u8, self.names),
            };
        }
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.slice);
            self.slice = undefined;
            self.allocator.free(self.shape);
            self.shape = undefined;
            self.allocator.free(self.names);
            self.names = undefined;
        }

        pub fn copy(self: *Self) !Self {
            return Self{
                .allocator = self.allocator,
                .slice = try self.allocator.dupe(T, self.slice),
                .shape = try self.allocator.dupe(usize, self.shape),
                .names = try self.allocator.dupe(?[]const u8, self.names),
            };
        }

        pub fn toString(self: Self) ![]u8 {
            const sliceChildType = util.childType(self.slice);
            const shapeString = try util.shapeToString(self.allocator, self.shape, self.names);
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

test "some tests" {
    const print = std.debug.print;
    _ = print;
    const allocator = testing.allocator;

    var iotaq = try Tensor(u32).iota(allocator, &.{ 2, 2, 3 }, .{ .names = &.{ "x", "y", "z" } });
    defer iotaq.deinit();
    try iotaq.debug_print();
    var prodq = try iotaq.product(.{ .axes = &.{0} });
    defer prodq.deinit();
    try prodq.debug_print();

    var sumq = try iotaq.sum(.{ .axes = &.{ 1, 0 } });
    defer sumq.deinit();
    try sumq.debug_print();

    var sumq2 = try iotaq.sum(.{});
    defer sumq2.deinit();
    try sumq2.debug_print();

    var iotaf = try Tensor(f64).iota(allocator, &.{ 2, 3, 4 }, .{});
    defer iotaf.deinit();

    var sumf = try iotaf.sum(.{ .axes = &.{1} });
    defer sumf.deinit();
    try sumf.debug_print();

    var prodf = try iotaf.product(.{ .axes = &.{1} });
    defer prodf.deinit();
    try prodf.debug_print();

    var alliota = try iotaq.all(.{ .axes = &.{0} });
    defer alliota.deinit();
    try alliota.debug_print();

    var anyiota = try iotaq.any(.{ .axes = &.{0} });
    defer anyiota.deinit();
    try anyiota.debug_print();

    var miniota = try iotaf.reduceMin(.{ .axes = &.{ 0, 2 } });
    defer miniota.deinit();
    try miniota.debug_print();

    var maxiota = try iotaf.reduceMax(.{ .axes = &.{ 0, 2 } });
    defer maxiota.deinit();
    try maxiota.debug_print();

    var meaniota = try iotaf.mean(.{ .axes = &.{ 0, 2 } });
    defer meaniota.deinit();
    try meaniota.debug_print();

    var i32Tensor = try Tensor(i32).tensor(allocator, &.{ -1, 0, 1, -2, -1, 0 }, .{ .shape = &.{ 2, 3 } });
    defer i32Tensor.deinit();

    var absi32 = try i32Tensor.abs();
    defer absi32.deinit();
    try absi32.debug_print();

    var usize_tensor = try Tensor(usize).tensor(allocator, &.{ 1, 2, 3, 4 }, .{ .shape = &.{ 2, 2 } });
    defer usize_tensor.deinit();
    var abs_usize = try usize_tensor.abs();
    defer abs_usize.deinit();
    try abs_usize.debug_print();

    var acoshiotaf = try iotaf.acosh();
    defer acoshiotaf.deinit();
    try acoshiotaf.debug_print();

    var acosiotaf = try iotaf.acos();
    defer acosiotaf.deinit();
    try acosiotaf.debug_print();

    var cosiotaf = try iotaf.cos();
    defer cosiotaf.deinit();
    try cosiotaf.debug_print();

    var f64_tensor = try Tensor(f64).tensor(allocator, &.{ 0.1, 0.2, 0.5, -0.4 }, .{ .shape = &.{ 2, 2 } });
    defer f64_tensor.deinit();
    var absf64 = try f64_tensor.abs();
    defer absf64.deinit();
    try absf64.debug_print();
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    _ = allocator;
}

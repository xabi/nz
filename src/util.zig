const std = @import("std");

pub fn abs(num: i32) i32 {
    if (num < 0) return -num;
    return num;
}

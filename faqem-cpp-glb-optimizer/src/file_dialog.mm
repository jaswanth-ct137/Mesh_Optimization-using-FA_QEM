#include "file_dialog.hpp"
#import <Cocoa/Cocoa.h>

std::filesystem::path choose_model_file() {
  @autoreleasepool {
    [NSApp activateIgnoringOtherApps:YES];
    NSOpenPanel *panel = [NSOpenPanel openPanel];
    [panel setCanChooseFiles:YES];
    [panel setCanChooseDirectories:NO];
    [panel setAllowsMultipleSelection:NO];
    [panel setAllowedFileTypes:@[@"glb", @"gltf", @"obj", @"stl", @"ply", @"off"]];
    [panel setTitle:@"Open model"];
    if ([panel runModal] != NSModalResponseOK) return {};
    return std::filesystem::path([[[panel URL] path] UTF8String]);
  }
}

void open_in_finder(const std::filesystem::path &path) {
  @autoreleasepool {
    NSString *p = [NSString stringWithUTF8String:path.string().c_str()];
    [[NSWorkspace sharedWorkspace] openURL:[NSURL fileURLWithPath:p]];
  }
}

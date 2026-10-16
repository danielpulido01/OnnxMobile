// Learn more https://docs.expo.dev/guides/customizing-metro
const { getDefaultConfig } = require("expo/metro-config");

const config = getDefaultConfig(__dirname);

// assets/model/model.onnx se empaqueta como asset (archivo binario), no como código.
config.resolver.assetExts.push("onnx");

module.exports = config;

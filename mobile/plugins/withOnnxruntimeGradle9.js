// onnxruntime-react-native 1.24.3 usa VersionNumber.parse() en su android/build.gradle, clase que Gradle 9
// eliminó ("Could not get unknown property 'VersionNumber'"). Corregido upstream en microsoft/onnxruntime#27385,
// sin versión publicada aún. Este plugin define un VersionNumber mínimo solo para ese subproyecto, en el
// android/build.gradle raíz que genera prebuild. Borrar cuando salga una versión con el arreglo.
const { withProjectBuildGradle } = require("expo/config-plugins");
const { mergeContents } = require("@expo/config-plugins/build/utils/generateCode");

const SHIM = `
// parse("0.86.3") -> 86003: enough for onnxruntime-react-native's "< 0.71" check.
class OrtVersionNumber {
  static long parse(String version) {
    def parts = version.tokenize('.-')
    return (0..2).sum { i -> (i < parts.size() && parts[i].isNumber() ? parts[i].toLong() : 0L) * (1000L ** (2 - i)) }
  }
}
subprojects { p ->
  if (p.name == 'onnxruntime-react-native') {
    p.ext.VersionNumber = OrtVersionNumber
  }
}`;

module.exports = function withOnnxruntimeGradle9(config) {
  return withProjectBuildGradle(config, (config) => {
    if (config.modResults.language !== "groovy") throw new Error("withOnnxruntimeGradle9: se esperaba build.gradle en Groovy");
    config.modResults.contents = mergeContents({
      src: config.modResults.contents,
      newSrc: SHIM,
      tag: "onnxruntime-gradle9-versionnumber",
      anchor: /apply plugin: "com\.facebook\.react\.rootproject"/,
      offset: 1,
      comment: "//",
    }).contents;
    return config;
  });
};

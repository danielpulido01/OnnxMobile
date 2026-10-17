// onnxruntime-react-native 1.24.3 no trae react-native.config.js, así que el autolinking no lo registra
// en Android (error "Cannot read property 'install' of null"). Corregido upstream en microsoft/onnxruntime#28266,
// pero aún sin versión publicada en npm: cuando salga, este archivo se puede borrar.
module.exports = {
  dependencies: {
    "onnxruntime-react-native": {
      platforms: {
        android: {
          packageImportPath: "import ai.onnxruntime.reactnative.OnnxruntimePackage;",
          packageInstance: "new OnnxruntimePackage()",
        },
      },
    },
  },
};

// Metro devuelve un id de asset para estos archivos (ver assetExts en metro.config.js).
declare module "*.onnx" {
  const asset: number;
  export default asset;
}

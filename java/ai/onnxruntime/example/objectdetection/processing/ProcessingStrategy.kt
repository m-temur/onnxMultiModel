package ai.onnxruntime.example.objectdetection.processing

import ai.onnxruntime.OnnxTensor
import android.graphics.Bitmap


interface ProcessingStrategy {
    fun preprocess(bitmap: Bitmap): OnnxTensor
    fun postprocess(outputs: Map<String, OnnxTensor>): DetectionResult
    fun getInputShape(): Array<Long>
    fun visualizeResult(originalBitmap: Bitmap, result: DetectionResult): Bitmap
}
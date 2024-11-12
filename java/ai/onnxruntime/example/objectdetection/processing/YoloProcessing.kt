package ai.onnxruntime.example.objectdetection.processing

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.example.objectdetection.models.DetectionModel
import android.graphics.Bitmap


class YoloProcessing(private val config: DetectionModel.Yolo) : ProcessingStrategy {
    override fun preprocess(bitmap: Bitmap): OnnxTensor {
        TODO("Not yet implemented")
    }

    override fun postprocess(outputs: Map<String, OnnxTensor>): DetectionResult {
        TODO("Not yet implemented")
    }

    override fun getInputShape() = arrayOf(1L, 3L,
        config.inputSize.first.toLong(),
        config.inputSize.second.toLong()
    )

    override fun visualizeResult(originalBitmap: Bitmap, result: DetectionResult): Bitmap {
        TODO("Not yet implemented")
    }
}
package ai.onnxruntime.example.objectdetection.detection

import android.graphics.Bitmap

/**
 * Extension of DetectorListener that includes callback for visualization
 */
interface VisualizationListener : DetectorListener {
    /**
     * Called when visualization of results is ready
     *
     * @param visualizedBitmap Bitmap containing the visualization of detection results
     */
    fun onVisualizationReady(visualizedBitmap: Bitmap)
}
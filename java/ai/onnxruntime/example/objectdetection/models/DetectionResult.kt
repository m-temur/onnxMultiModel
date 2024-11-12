package ai.onnxruntime.example.objectdetection.models

import android.os.Parcelable
import kotlinx.android.parcel.Parcelize

@Parcelize
data class DetectionResult(
    val score: Float,
    val label: String,
    val heatmap: FloatArray,
    val threshold: Float,
    val confidence: Float
) : Parcelable {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as DetectionResult

        if (score != other.score) return false
        if (label != other.label) return false
        if (!heatmap.contentEquals(other.heatmap)) return false
        if (threshold != other.threshold) return false
        if (confidence != other.confidence) return false

        return true
    }

    override fun hashCode(): Int {
        var result = score.hashCode()
        result = 31 * result + label.hashCode()
        result = 31 * result + heatmap.contentHashCode()
        result = 31 * result + threshold.hashCode()
        result = 31 * result + confidence.hashCode()
        return result
    }

    companion object {
        const val LABEL_NORMAL = "Normal"
        const val LABEL_ANOMALOUS = "Anomalous"
    }
}
using Microsoft.ML.Runtime.Api;

namespace TrainingModel
{
    // STEP 1: 定義資料模型
    //         MachineStatusData 資料模型用於訓練資料使用，並可作為預測資料模型
    //         - 前 4 個屬性為輸入的特性值，用來預測 Label 標籤
    //         - Label 標籤是我們要預測的屬性，只有在訓練資料時，才會主動提供值
    public class MachineStatusData
    {
        [Column("0")]
        public float MachineTemperature;

        [Column("1")]
        public float MachinePressure;

        [Column("2")]
        public float AmbientTemperature;

        [Column("3")]
        public float AmbientHumidity;

        [Column("4")]
        public string Label;
    }

    // STEP 1: 定義資料模型
    //         MachineStatusPrediction 是執行預測後的結果資料模型
    public class MachineStatusPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabel;
    }
}

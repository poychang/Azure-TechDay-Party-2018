using System.Collections.Generic;
using System.Threading.Tasks;

namespace TrainingModel
{
    public class Program
    {
        public static async Task Main(string[] args)
        {
            Prediction.Training();

            var samples = new List<MachineStatusData>()
            {
                new MachineStatusData() {
                    Label = "",
                    MachineTemperature = 101.52f,
                    MachinePressure =     10.28f,
                    AmbientTemperature =  20.87f,
                    AmbientHumidity =     26.00f,
                },
                new MachineStatusData()
                {
                    Label = "",
                    MachineTemperature = 103.65f,
                    MachinePressure =     10.45f,
                    AmbientTemperature =  20.90f,
                    AmbientHumidity =     26.00f,
                }
            };
            foreach (var sample in samples)
            {
                await Prediction.PredictFromFile(sample);
            }
        }
    }
}

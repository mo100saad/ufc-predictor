import React from 'react';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { Doughnut } from 'react-chartjs-2';

// Register Chart.js components
ChartJS.register(ArcElement, Tooltip, Legend);

const PredictionResult = ({ result }) => {
  const { fighter1_name, fighter2_name, fighter1_win_probability, fighter2_win_probability, predicted_winner, confidence_level } = result;
  
  // Prepare chart data
  const chartData = {
    labels: [fighter1_name, fighter2_name],
    datasets: [
      {
        data: [fighter1_win_probability * 100, fighter2_win_probability * 100],
        backgroundColor: ['#D20A0A', '#1277BC'],
        borderColor: ['#D20A0A', '#1277BC'],
        borderWidth: 1,
      },
    ],
  };
  
  // Chart options
  const chartOptions = {
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          color: 'white',
          font: {
            size: 14,
          },
        },
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            return `${context.label}: ${context.raw.toFixed(1)}%`;
          }
        }
      }
    },
    cutout: '70%'
  };
  
  // Determine confidence class color
  const confidenceColor = 
    confidence_level === 'High' ? 'text-green-500' :
    confidence_level === 'Medium' ? 'text-yellow-500' :
    'text-gray-400';

  return (
    <div className="bg-card-bg rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-bold mb-4">Prediction Results</h2>
      
      <div className="text-center mb-6">
        <div className="mb-2">
          <span className={predicted_winner === "fighter1" ? "text-ufc-red font-bold text-xl" : "text-xl"}>
            {fighter1_name}
          </span>
          <span className="text-xl mx-3 text-gray-400">vs</span>
          <span className={predicted_winner === "fighter2" ? "text-ufc-blue font-bold text-xl" : "text-xl"}>
            {fighter2_name}
          </span>
        </div>
      </div>
      
      <div className="h-64 mb-4">
        <Doughnut data={chartData} options={chartOptions} />
      </div>
      
      <div className="mt-6 bg-gray-800 rounded-lg p-4">
        <h3 className="text-xl font-bold mb-2">Prediction Summary</h3>
        <p className="mb-2">
          <span className="text-gray-400">Predicted Winner: </span>
          <span className={predicted_winner === "fighter1" ? "text-ufc-red font-medium" : "text-ufc-blue font-medium"}>
            {predicted_winner === "fighter1" ? fighter1_name : fighter2_name}
          </span>
        </p>
        <p className="mb-2">
          <span className="text-gray-400">Win Probability: </span>
          <span className="font-medium">
            {((predicted_winner === "fighter1" ? fighter1_win_probability : fighter2_win_probability) * 100).toFixed(1)}%
          </span>
        </p>
        <p>
          <span className="text-gray-400">Confidence: </span>
          <span className={`font-medium ${confidenceColor}`}>{confidence_level}</span>
        </p>
      </div>
    </div>
  );
};

export default PredictionResult;
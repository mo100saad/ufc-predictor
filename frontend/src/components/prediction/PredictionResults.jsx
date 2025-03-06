import React from 'react';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { Doughnut } from 'react-chartjs-2';
import { motion } from 'framer-motion';

// Register Chart.js components
ChartJS.register(ArcElement, Tooltip, Legend);

const PredictionResult = ({ result, fighter1, fighter2 }) => {
  // Extract backend probabilities
  const {
    fighter1_name,
    fighter2_name,
    fighter1_win_probability,
    fighter2_win_probability,
    confidence_level
    // ignoring "predicted_winner" to avoid mismatches
  } = result;

  // Convert probabilities to integer percentages
  const fighter1Percent = Math.round(fighter1_win_probability * 100);
  const fighter2Percent = Math.round(fighter2_win_probability * 100);

  // Determine final winner based on which percentage is higher
  const finalWinner = fighter1Percent > fighter2Percent ? 'fighter1' : 'fighter2';
  const finalWinnerName = finalWinner === 'fighter1' ? fighter1_name : fighter2_name;

  // Prepare chart data
  const chartData = {
    labels: [fighter1_name, fighter2_name],
    datasets: [
      {
        data: [fighter1Percent, fighter2Percent],
        backgroundColor: ['rgba(210, 10, 10, 0.8)', 'rgba(18, 119, 188, 0.8)'],
        borderColor: ['rgba(210, 10, 10, 1)', 'rgba(18, 119, 188, 1)'],
        borderWidth: 2,
      },
    ],
  };

  // Chart options
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'bottom',
        labels: {
          color: 'white',
          usePointStyle: true,
        },
      },
      tooltip: {
        enabled: false,
      },
    },
    cutout: '70%',
    animation: false,
  };

  return (
    <div className="bg-[#0F1729] text-white p-6 rounded-lg">
      {/* Header */}
      <div className="flex justify-center items-center space-x-3 mb-6">
        <div
          className={`text-xl font-bold ${
            finalWinner === 'fighter1' ? 'text-red-500' : 'text-gray-300'
          }`}
        >
          {fighter1_name}
        </div>
        <div className="text-xl px-3 py-1 bg-gray-800 rounded-full text-gray-400 font-medium text-sm">
          VS
        </div>
        <div
          className={`text-xl font-bold ${
            finalWinner === 'fighter2' ? 'text-blue-500' : 'text-gray-300'
          }`}
        >
          {fighter2_name}
        </div>
      </div>

      {/* Stats Row */}
      <div className="flex justify-between text-sm mb-4">
        <div className="text-left">
          <div>Record: {fighter1.wins}-{fighter1.losses}</div>
          <div>Striking: {fighter1.SLpM.toFixed(1)}/min</div>
          <div>TD Avg: {fighter1.td_avg.toFixed(1)}</div>
          <div>Defense: {(fighter1.str_def * 100).toFixed(0)}%</div>
        </div>
        <div className="text-right">
          <div>Record: {fighter2.wins}-{fighter2.losses}</div>
          <div>Striking: {fighter2.SLpM.toFixed(1)}/min</div>
          <div>TD Avg: {fighter2.td_avg.toFixed(1)}</div>
          <div>Defense: {(fighter2.str_def * 100).toFixed(0)}%</div>
        </div>
      </div>

      {/* Chart Container */}
      <div className="relative h-60 mb-4">
        <Doughnut data={chartData} options={chartOptions} />
      </div>

      {/* Winner Label - moved below the chart */}
      <div className="text-center mb-4">
        <div className="text-xs uppercase text-gray-400 mb-1">Winner</div>
        <div
          className={`font-bold ${
            finalWinner === 'fighter1' ? 'text-red-500' : 'text-blue-500'
          }`}
        >
          {finalWinnerName}
        </div>
      </div>

      {/* Prediction Analysis */}
      <div className="bg-[#1E293B] rounded-lg p-4">
        <div className="text-purple-400 text-lg font-bold mb-4">Prediction Analysis</div>
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-4xl font-bold text-blue-500">{fighter1Percent}%</div>
            <div className="text-xs text-gray-400">Win Probability</div>
          </div>
          <div>
            <div className="text-xl font-bold text-yellow-500">{confidence_level}</div>
            <div className="text-xs text-gray-400">Prediction Confidence</div>
          </div>
          <div>
            <div className="text-4xl font-bold text-gray-300">
              {Math.abs(fighter1Percent - fighter2Percent)}%
            </div>
            <div className="text-xs text-gray-400">Victory Margin</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionResult;

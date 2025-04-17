import React from 'react';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { Doughnut } from 'react-chartjs-2';
import { motion } from 'framer-motion';
import FighterImage from '../fighters/FighterImage';

// Register Chart.js components
ChartJS.register(ArcElement, Tooltip, Legend);

const PredictionResult = ({ result, fighter1, fighter2 }) => {
  // Extract backend probabilities
  const {
    fighter1_name,
    fighter2_name,
    fighter1_win_probability,
    fighter2_win_probability,
    confidence_level,
    fighter1_image_url,
    fighter2_image_url
    // ignoring "predicted_winner" to avoid mismatches
  } = result;

  // Convert probabilities to integer percentages
  const fighter1Percent = Math.round(fighter1_win_probability * 100);
  const fighter2Percent = Math.round(fighter2_win_probability * 100);

  // Determine final winner based on which percentage is higher
  const finalWinner = fighter1Percent > fighter2Percent ? 'fighter1' : 'fighter2';
  const finalWinnerName = finalWinner === 'fighter1' ? fighter1_name : fighter2_name;
  const finalWinnerData = finalWinner === 'fighter1' ? fighter1 : fighter2;
  const loserData = finalWinner === 'fighter1' ? fighter2 : fighter1;

  // Calculate fighter advantages
  const getAdvantageText = () => {
    let advantages = [];
    
    // Striking advantage
    if ((fighter1.SLpM || 0) > (fighter2.SLpM || 0) + 1) {
      advantages.push(`${fighter1_name} has superior striking volume (${fighter1.SLpM?.toFixed(1)} vs ${fighter2.SLpM?.toFixed(1)} strikes/min)`);
    } else if ((fighter2.SLpM || 0) > (fighter1.SLpM || 0) + 1) {
      advantages.push(`${fighter2_name} has superior striking volume (${fighter2.SLpM?.toFixed(1)} vs ${fighter1.SLpM?.toFixed(1)} strikes/min)`);
    }
    
    // Takedown advantage
    if ((fighter1.td_avg || 0) > (fighter2.td_avg || 0) + 1) {
      advantages.push(`${fighter1_name} has superior takedown ability (${fighter1.td_avg?.toFixed(1)} vs ${fighter2.td_avg?.toFixed(1)} per fight)`);
    } else if ((fighter2.td_avg || 0) > (fighter1.td_avg || 0) + 1) {
      advantages.push(`${fighter2_name} has superior takedown ability (${fighter2.td_avg?.toFixed(1)} vs ${fighter1.td_avg?.toFixed(1)} per fight)`);
    }
    
    // Defense advantage
    if ((fighter1.str_def || 0) > (fighter2.str_def || 0) + 0.1) {
      advantages.push(`${fighter1_name} has better striking defense (${(fighter1.str_def * 100).toFixed(0)}% vs ${(fighter2.str_def * 100).toFixed(0)}%)`);
    } else if ((fighter2.str_def || 0) > (fighter1.str_def || 0) + 0.1) {
      advantages.push(`${fighter2_name} has better striking defense (${(fighter2.str_def * 100).toFixed(0)}% vs ${(fighter1.str_def * 100).toFixed(0)}%)`);
    }
    
    // Experience advantage
    const f1Fights = (fighter1.wins || 0) + (fighter1.losses || 0);
    const f2Fights = (fighter2.wins || 0) + (fighter2.losses || 0);
    if (f1Fights > f2Fights + 5) {
      advantages.push(`${fighter1_name} has more experience (${f1Fights} fights vs ${f2Fights} fights)`);
    } else if (f2Fights > f1Fights + 5) {
      advantages.push(`${fighter2_name} has more experience (${f2Fights} fights vs ${f1Fights} fights)`);
    }
    
    // Win rate advantage
    const f1WinRate = f1Fights > 0 ? (fighter1.wins || 0) / f1Fights : 0;
    const f2WinRate = f2Fights > 0 ? (fighter2.wins || 0) / f2Fights : 0;
    if (f1WinRate > f2WinRate + 0.1 && f1Fights > 5) {
      advantages.push(`${fighter1_name} has a better win rate (${(f1WinRate * 100).toFixed(0)}% vs ${(f2WinRate * 100).toFixed(0)}%)`);
    } else if (f2WinRate > f1WinRate + 0.1 && f2Fights > 5) {
      advantages.push(`${fighter2_name} has a better win rate (${(f2WinRate * 100).toFixed(0)}% vs ${(f1WinRate * 100).toFixed(0)}%)`);
    }
    
    return advantages.length > 0 ? advantages : [`This appears to be a closely matched fight based on the stats.`];
  };

  // Chart aesthetics - improved with gradients and better visualization
  const chartData = {
    labels: [fighter1_name, fighter2_name],
    datasets: [
      {
        data: [fighter1Percent, fighter2Percent],
        backgroundColor: [
          'rgba(220, 38, 38, 0.8)', // Red corner - deeper red
          'rgba(37, 99, 235, 0.8)'  // Blue corner - deeper blue
        ],
        borderColor: [
          'rgba(248, 113, 113, 1)', // Lighter red border
          'rgba(96, 165, 250, 1)'   // Lighter blue border
        ],
        borderWidth: 2,
        hoverBackgroundColor: [
          'rgba(220, 38, 38, 0.9)',
          'rgba(37, 99, 235, 0.9)'
        ],
        hoverBorderColor: [
          'rgba(248, 113, 113, 1)',
          'rgba(96, 165, 250, 1)'
        ],
        hoverBorderWidth: 3,
      },
    ],
  };

  // Enhanced chart options with larger size and optimized layout
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
          padding: 20,
          font: {
            size: 12,
            weight: 'bold'
          }
        },
      },
      tooltip: {
        enabled: true,
        backgroundColor: 'rgba(15, 23, 42, 0.9)',
        titleFont: {
          size: 14,
          weight: 'bold'
        },
        bodyFont: {
          size: 12
        },
        padding: 10,
        callbacks: {
          label: function(context) {
            return context.label + ': ' + context.raw + '% win probability';
          }
        }
      },
    },
    cutout: '70%', // Slightly smaller cutout to make ring larger
    animation: {
      animateScale: true,
      animateRotate: true,
      duration: 1000,
      easing: 'easeOutQuart'
    },
    rotation: -90, // Start from top
    circumference: 360, // Full circle
  };

  // Animation variants for content
  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };
  
  const item = {
    hidden: { y: 20, opacity: 0 },
    show: { y: 0, opacity: 1 }
  };

  return (
    <motion.div 
      className="backdrop-blur-sm bg-gradient-to-br from-gray-900/90 to-gray-800/90 text-white p-6 rounded-lg shadow-xl border border-gray-700"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      {/* Header with Fighter Images */}
      <motion.div 
        className="flex justify-between items-center mb-6"
        variants={container}
        initial="hidden"
        animate="show"
      >
        <motion.div variants={item} className="flex flex-col items-center">
          <div className={`overflow-hidden ${finalWinner === 'fighter1' ? 'shadow-xl shadow-red-500/50' : ''}`}>
            {/* MUCH larger fighter image on prediction page */}
            <FighterImage 
              src={fighter1_image_url} 
              alt={fighter1_name}
              size="xl" 
              rounded={false}
              borderColor={finalWinner === 'fighter1' ? 'border-red-500' : 'border-gray-600'}
              className="transform scale-110 hover:scale-115 transition-transform duration-300"
            />
          </div>
          <div className={`text-xl font-bold mt-3 ${finalWinner === 'fighter1' ? 'text-red-500' : 'text-gray-300'}`}>
            {fighter1_name}
          </div>
          <div className="text-sm text-gray-400 mt-1">
            Record: {fighter1.wins}-{fighter1.losses}
          </div>
        </motion.div>

        <motion.div variants={item} className="text-2xl px-8 py-3 bg-gray-800/70 backdrop-blur-sm rounded-full text-gray-200 font-medium mx-6 border border-gray-700/50 shadow-lg">
          VS
        </motion.div>

        <motion.div variants={item} className="flex flex-col items-center">
          <div className={`overflow-hidden ${finalWinner === 'fighter2' ? 'shadow-xl shadow-blue-500/50' : ''}`}>
            {/* MUCH larger fighter image on prediction page */}
            <FighterImage 
              src={fighter2_image_url} 
              alt={fighter2_name}
              size="xl"
              rounded={false}
              borderColor={finalWinner === 'fighter2' ? 'border-blue-500' : 'border-gray-600'}
              className="transform scale-110 hover:scale-115 transition-transform duration-300"
            />
          </div>
          <div className={`text-xl font-bold mt-3 ${finalWinner === 'fighter2' ? 'text-blue-500' : 'text-gray-300'}`}>
            {fighter2_name}
          </div>
          <div className="text-sm text-gray-400 mt-1">
            Record: {fighter2.wins}-{fighter2.losses}
          </div>
        </motion.div>
      </motion.div>

      {/* Main Content Grid - Increased vertical spacing to accommodate larger fighter images */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-8">
        {/* Left Column - Stat Comparison */}
        <motion.div 
          className="bg-gray-800/50 rounded-lg p-4"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3, duration: 0.5 }}
        >
          <h3 className="text-base font-semibold text-gray-300 mb-3 uppercase tracking-wider">Fighter Comparison</h3>
          
          {/* Stat Comparison Table */}
          <table className="w-full text-sm">
            <tbody>
              {/* Striking Stats */}
              <tr className="border-b border-gray-700/50">
                <td className="py-2 text-right pr-2">
                  <span className={`px-1.5 py-0.5 rounded ${fighter1.SLpM > fighter2.SLpM ? 'bg-red-900/40 text-red-400' : 'text-gray-400'}`}>
                    {fighter1.SLpM?.toFixed(1) || 'N/A'}
                  </span>
                </td>
                <td className="py-2 text-center text-xs text-gray-500">Strikes/Min</td>
                <td className="py-2 text-left pl-2">
                  <span className={`px-1.5 py-0.5 rounded ${fighter2.SLpM > fighter1.SLpM ? 'bg-blue-900/40 text-blue-400' : 'text-gray-400'}`}>
                    {fighter2.SLpM?.toFixed(1) || 'N/A'}
                  </span>
                </td>
              </tr>
              
              {/* Takedown Stats */}
              <tr className="border-b border-gray-700/50">
                <td className="py-2 text-right pr-2">
                  <span className={`px-1.5 py-0.5 rounded ${fighter1.td_avg > fighter2.td_avg ? 'bg-red-900/40 text-red-400' : 'text-gray-400'}`}>
                    {fighter1.td_avg?.toFixed(1) || 'N/A'}
                  </span>
                </td>
                <td className="py-2 text-center text-xs text-gray-500">Takedowns/Fight</td>
                <td className="py-2 text-left pl-2">
                  <span className={`px-1.5 py-0.5 rounded ${fighter2.td_avg > fighter1.td_avg ? 'bg-blue-900/40 text-blue-400' : 'text-gray-400'}`}>
                    {fighter2.td_avg?.toFixed(1) || 'N/A'}
                  </span>
                </td>
              </tr>
              
              {/* Striking Accuracy */}
              <tr className="border-b border-gray-700/50">
                <td className="py-2 text-right pr-2">
                  <span className={`px-1.5 py-0.5 rounded ${fighter1.sig_str_acc > fighter2.sig_str_acc ? 'bg-red-900/40 text-red-400' : 'text-gray-400'}`}>
                    {fighter1.sig_str_acc ? (fighter1.sig_str_acc * 100).toFixed(0) : 'N/A'}%
                  </span>
                </td>
                <td className="py-2 text-center text-xs text-gray-500">Strike Accuracy</td>
                <td className="py-2 text-left pl-2">
                  <span className={`px-1.5 py-0.5 rounded ${fighter2.sig_str_acc > fighter1.sig_str_acc ? 'bg-blue-900/40 text-blue-400' : 'text-gray-400'}`}>
                    {fighter2.sig_str_acc ? (fighter2.sig_str_acc * 100).toFixed(0) : 'N/A'}%
                  </span>
                </td>
              </tr>
              
              {/* Defense */}
              <tr>
                <td className="py-2 text-right pr-2">
                  <span className={`px-1.5 py-0.5 rounded ${fighter1.str_def > fighter2.str_def ? 'bg-red-900/40 text-red-400' : 'text-gray-400'}`}>
                    {fighter1.str_def ? (fighter1.str_def * 100).toFixed(0) : 'N/A'}%
                  </span>
                </td>
                <td className="py-2 text-center text-xs text-gray-500">Defense</td>
                <td className="py-2 text-left pl-2">
                  <span className={`px-1.5 py-0.5 rounded ${fighter2.str_def > fighter1.str_def ? 'bg-blue-900/40 text-blue-400' : 'text-gray-400'}`}>
                    {fighter2.str_def ? (fighter2.str_def * 100).toFixed(0) : 'N/A'}%
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </motion.div>
        
        {/* Right Column - Chart */}
        <motion.div 
          className="relative min-h-[250px]"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3, duration: 0.5 }}
        >
          <Doughnut data={chartData} options={chartOptions} />
          
          {/* Center winner label - Repositioned to avoid overlap */}
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center pointer-events-none">
            <div className="bg-gray-800/80 rounded-lg px-4 py-3 backdrop-blur-sm shadow-lg border border-gray-700/50 max-w-[80%]">
              <div className="text-xs uppercase text-gray-400 mb-1">Predicted Winner</div>
              <div className={`text-xl font-bold ${finalWinner === 'fighter1' ? 'text-red-500' : 'text-blue-500'}`}>
                {finalWinnerName}
              </div>
              <div className="mt-1 text-xs text-gray-400">
                {Math.abs(fighter1Percent - fighter2Percent)}% margin
              </div>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Prediction Analysis Panel */}
      <motion.div 
        className="mt-6 bg-gray-800/70 rounded-lg p-5 border border-gray-700/50"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5, duration: 0.5 }}
      >
        <div className="flex items-center mb-4">
          <div className="bg-purple-900/30 p-1.5 rounded-md mr-3">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-purple-400" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
          </div>
          <h3 className="text-purple-400 text-lg font-bold">AI Analysis</h3>
          <div className={`ml-auto px-2 py-1 rounded text-xs font-medium ${
            confidence_level === 'High' ? 'bg-green-900/40 text-green-400' :
            confidence_level === 'Medium' ? 'bg-yellow-900/40 text-yellow-400' :
            'bg-red-900/40 text-red-400'
          }`}>
            {confidence_level} Confidence
          </div>
        </div>
        
        {/* Key Advantages */}
        <div className="mb-4">
          <div className="text-sm text-gray-300 mb-2 font-medium">Key Factors:</div>
          <ul className="text-xs text-gray-400 space-y-1.5">
            {getAdvantageText().map((advantage, index) => (
              <li key={index} className="flex items-start">
                <span className="inline-block w-4 h-4 mr-2 mt-0.5 text-indigo-400">•</span>
                <span>{advantage}</span>
              </li>
            ))}
          </ul>
        </div>
        
        {/* Win Probability Meters */}
        <div className="grid grid-cols-2 gap-4 mt-5">
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-400">{fighter1_name}</span>
              <span className="text-red-400 font-medium">{fighter1Percent}%</span>
            </div>
            <div className="w-full bg-gray-700/50 rounded-full h-2.5">
              <div 
                className="bg-gradient-to-r from-red-800 to-red-500 h-2.5 rounded-full" 
                style={{ width: `${fighter1Percent}%` }}
              ></div>
            </div>
          </div>
          
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-400">{fighter2_name}</span>
              <span className="text-blue-400 font-medium">{fighter2Percent}%</span>
            </div>
            <div className="w-full bg-gray-700/50 rounded-full h-2.5">
              <div 
                className="bg-gradient-to-r from-blue-800 to-blue-500 h-2.5 rounded-full" 
                style={{ width: `${fighter2Percent}%` }}
              ></div>
            </div>
          </div>
        </div>
        
        {/* Disclaimer */}
        <div className="mt-5 pt-3 border-t border-gray-700/40 text-[10px] text-gray-500 italic text-center">
          This prediction is based on historical fighter statistics and mathematical models. 
          UFC fights are highly unpredictable and results can vary.
        </div>
      </motion.div>
    </motion.div>
  );
};

export default PredictionResult;

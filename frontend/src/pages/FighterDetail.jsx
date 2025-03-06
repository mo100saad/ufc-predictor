import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { fighterService } from '../services/api';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorAlert from '../components/common/ErrorAlert';
import { motion } from 'framer-motion';
import { Chart as ChartJS, RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend } from 'chart.js';
import { Radar } from 'react-chartjs-2';

// Register Chart.js components
ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend);

const FighterDetail = () => {
  const { name } = useParams();
  const [fighter, setFighter] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  useEffect(() => {
    const loadFighter = async () => {
      try {
        setLoading(true);
        console.log(`Attempting to load fighter: ${name}`);
        
        const data = await fighterService.getFighterByName(name);
        
        console.log('Received fighter data:', data);
        
        if (!data) {
          console.error(`No data found for fighter: ${name}`);
          setError(`No data found for fighter ${name}`);
          setLoading(false);
          return;
        }
        
        setFighter(data);
        setLoading(false);
      } catch (err) {
        console.error('Complete error:', err);
        
        // Check if it's a 404 error
        if (err.response && err.response.status === 404) {
          setError(`Fighter not found: ${name}`);
        } else {
          setError(`Failed to load data for ${name}. ${err.message}`);
        }
        
        setLoading(false);
      }
    };
  
    loadFighter();
  }, [name]);

  // Animation variants
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
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 }
  };

  // Create radar chart data
  const getRadarData = (fighter) => {
    if (!fighter) return null;

    // Normalize values to 0-100 scale
    const normalize = (value, max) => Math.min(100, Math.max(0, (value / max) * 100));

    return {
      labels: ['Striking', 'Takedowns', 'Submissions', 'Defense', 'Experience', 'Power'],
      datasets: [
        {
          label: 'Fighter Stats',
          data: [
            normalize(fighter.SLpM || 0, 8),                    // Striking (max ~8 strikes per min)
            normalize(fighter.td_avg || 0, 6),                  // Takedowns (max ~6 per fight)
            normalize(fighter.sub_avg || 0, 2),                 // Submissions (max ~2 per fight)
            normalize((fighter.str_def || 0.5) * 100, 100),     // Defense (percentage)
            normalize(fighter.wins || 0, 30),                   // Experience (max ~30 wins)
            normalize(fighter.SLpM * (fighter.sig_str_acc || 0.5), 4)  // Power - combination of strikes and accuracy
          ],
          backgroundColor: 'rgba(255, 99, 132, 0.2)',
          borderColor: 'rgba(255, 99, 132, 1)',
          borderWidth: 2,
          pointBackgroundColor: 'rgba(255, 99, 132, 1)',
          pointBorderColor: '#fff',
          pointHoverBackgroundColor: '#fff',
          pointHoverBorderColor: 'rgba(255, 99, 132, 1)',
          pointRadius: 4
        }
      ]
    };
  };

  const radarOptions = {
    scales: {
      r: {
        angleLines: {
          color: 'rgba(255, 255, 255, 0.1)'
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        },
        pointLabels: {
          color: 'rgba(255, 255, 255, 0.7)',
          font: {
            size: 12
          }
        },
        ticks: {
          backdropColor: 'transparent',
          color: 'rgba(255, 255, 255, 0.5)',
          showLabelBackdrop: false
        },
        suggestedMin: 0,
        suggestedMax: 100
      }
    },
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: 'rgba(255, 99, 132, 1)',
        bodyColor: '#fff',
        cornerRadius: 6,
        padding: 10
      }
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <LoadingSpinner text={`Loading data for ${name}...`} />
      </div>
    );
  }

  if (error) {
    return (
      <div>
        <ErrorAlert message={error} />
        <Link to="/fighters" className="text-blue-500 hover:text-red-500 transition-colors duration-300 mt-4 inline-block">
          ← Back to fighters
        </Link>
      </div>
    );
  }

  if (!fighter) {
    return (
      <div className="text-center py-10">
        <p className="text-gray-400">Fighter not found</p>
        <Link to="/fighters" className="text-blue-500 hover:text-red-500 transition-colors duration-300 mt-4 inline-block">
          ← Back to fighters
        </Link>
      </div>
    );
  }

  // Calculate fight style based on stats
  const getFightStyle = () => {
    if (!fighter) return "Unknown";
    
    const strikeRatio = (fighter.SLpM || 0) / 5; // Normalized to 0-1 where 5 is high
    const grapplingRatio = ((fighter.td_avg || 0) + (fighter.sub_avg || 0)) / 6; // Normalized
    
    if (strikeRatio > 0.7 && grapplingRatio < 0.3) return "Pure Striker";
    if (strikeRatio < 0.3 && grapplingRatio > 0.7) return "Dominant Grappler";
    if (strikeRatio > 0.5 && grapplingRatio > 0.5) return "Well-Rounded";
    if (strikeRatio > 0.5) return "Striker";
    if (grapplingRatio > 0.5) return "Grappler";
    return "Balanced Fighter";
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div className="mb-6">
        <Link to="/fighters" className="text-blue-500 hover:text-red-500 transition-colors duration-300 flex items-center">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Back to fighters
        </Link>
      </div>
      
      <motion.div 
        variants={container}
        initial="hidden"
        animate="show"
        className="grid grid-cols-1 md:grid-cols-3 gap-8"
      >
        {/* Fighter Profile - 1 column */}
        <motion.div variants={item} className="md:col-span-1">
          <div className="card backdrop-blur-sm bg-gray-900/60 border border-gray-800 shadow-xl rounded-lg overflow-hidden">
            {/* Fighter Name Header */}
            <div className="bg-gradient-to-r from-gray-800 to-gray-900 p-6 border-b border-gray-700">
              <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-red-400 to-blue-400">{fighter.name}</h1>
              <div className="text-lg text-gray-300 mt-2 flex items-center">
                <span className="bg-gray-800 px-2 py-1 rounded text-sm font-medium mr-2">
                  {fighter.wins || 0}-{fighter.losses || 0}{fighter.draws ? `-${fighter.draws}` : ''}
                </span>
                <span className="text-gray-500 text-sm">{getFightStyle()}</span>
              </div>
            </div>
            
            {/* Physical Attributes */}
            <div className="p-6">
              <h2 className="text-xl font-bold mb-4 text-gray-200">Physical Attributes</h2>
              <div className="space-y-3">
                {[
                  { icon: '📏', label: 'Height', value: fighter.height ? `${fighter.height} cm` : 'N/A' },
                  { icon: '⚖️', label: 'Weight', value: fighter.weight ? `${fighter.weight} lbs` : 'N/A' },
                  { icon: '🤜', label: 'Reach', value: fighter.reach ? `${fighter.reach} cm` : 'N/A' },
                  { icon: '🥋', label: 'Stance', value: fighter.stance || 'N/A' }
                ].map((attr, index) => (
                  <div key={index} className="flex items-center justify-between bg-gray-800/50 p-3 rounded-lg">
                    <div className="flex items-center">
                      <span className="mr-3">{attr.icon}</span>
                      <span className="text-gray-400">{attr.label}</span>
                    </div>
                    <span className="font-medium text-white">{attr.value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          
          {/* Win Breakdown */}
          <motion.div variants={item} className="mt-8">
            <div className="card backdrop-blur-sm bg-gray-900/60 border border-gray-800 shadow-xl rounded-lg p-6">
              <h2 className="text-xl font-bold mb-4 text-gray-200">Win Breakdown</h2>
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-gray-800/50 p-4 rounded-lg text-center">
                  <div className="text-3xl font-bold text-red-500 mb-1">{fighter.win_by_KO_TKO || '0'}</div>
                  <div className="text-xs text-gray-400">KO/TKO</div>
                </div>
                <div className="bg-gray-800/50 p-4 rounded-lg text-center">
                  <div className="text-3xl font-bold text-blue-500 mb-1">{fighter.win_by_SUB || '0'}</div>
                  <div className="text-xs text-gray-400">Submission</div>
                </div>
                <div className="bg-gray-800/50 p-4 rounded-lg text-center">
                  <div className="text-3xl font-bold text-yellow-500 mb-1">{fighter.win_by_DEC || '0'}</div>
                  <div className="text-xs text-gray-400">Decision</div>
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>
        
        {/* Stats & Visualization - 2 columns */}
        <motion.div variants={item} className="md:col-span-2">
          <div className="card backdrop-blur-sm bg-gray-900/60 border border-gray-800 shadow-xl rounded-lg p-6">
            <h2 className="text-xl font-bold mb-6 text-gray-200">Performance Analysis</h2>
            
            <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
              {/* Stats - 2 columns */}
              <div className="lg:col-span-2">
                <h3 className="text-lg font-medium mb-4 text-gray-300">Fighting Statistics</h3>
                <div className="space-y-4">
                  {[
                    { label: 'Striking Rate', value: fighter.SLpM ? `${fighter.SLpM.toFixed(1)}/min` : 'N/A', bg: 'bg-red-900/20' },
                    { label: 'Striking Accuracy', value: fighter.sig_str_acc ? `${(fighter.sig_str_acc * 100).toFixed(0)}%` : 'N/A', bg: 'bg-red-900/20' },
                    { label: 'Striking Defense', value: fighter.str_def ? `${(fighter.str_def * 100).toFixed(0)}%` : 'N/A', bg: 'bg-blue-900/20' },
                    { label: 'Takedown Average', value: fighter.td_avg ? `${fighter.td_avg.toFixed(1)}/fight` : 'N/A', bg: 'bg-green-900/20' },
                    { label: 'Takedown Defense', value: fighter.td_def ? `${(fighter.td_def * 100).toFixed(0)}%` : 'N/A', bg: 'bg-green-900/20' },
                    { label: 'Submission Average', value: fighter.sub_avg ? `${fighter.sub_avg.toFixed(1)}/fight` : 'N/A', bg: 'bg-purple-900/20' }
                  ].map((stat, index) => (
                    <div key={index} className={`flex justify-between items-center p-3 rounded-lg ${stat.bg}`}>
                      <span className="text-gray-300">{stat.label}</span>
                      <span className="font-medium text-white">{stat.value}</span>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Radar Chart - 3 columns */}
              <div className="lg:col-span-3 flex flex-col justify-center">
                <h3 className="text-lg font-medium mb-4 text-gray-300 text-center">Fighter Radar</h3>
                <div className="h-64">
                  {getRadarData(fighter) && (
                    <Radar data={getRadarData(fighter)} options={radarOptions} />
                  )}
                </div>
              </div>
            </div>
            
            {/* Fighter strength analysis */}
            <div className="mt-8">
              <h3 className="text-lg font-medium mb-4 text-gray-300">Fighter Analysis</h3>
              <div className="bg-gray-800/50 p-4 rounded-lg">
                <p className="text-gray-300 mb-3">
                  {fighter.name} is a {getFightStyle().toLowerCase()} with {fighter.wins || 0} professional wins.
                  {fighter.SLpM > 4 ? ` Known for high striking output (${fighter.SLpM.toFixed(1)} strikes/min).` : ''}
                  {fighter.td_avg > 3 ? ` Excellent takedown ability (${fighter.td_avg.toFixed(1)} per fight).` : ''}
                  {fighter.sub_avg > 1 ? ` Submission specialist (${fighter.sub_avg.toFixed(1)} per fight).` : ''}
                  {fighter.td_def > 0.8 ? ` Exceptional takedown defense (${(fighter.td_def * 100).toFixed(0)}%).` : ''}
                </p>
                <p className="text-gray-300">
                  {fighter.SLpM > fighter.SApM ? 
                    `Offensive fighter who lands more strikes (${fighter.SLpM?.toFixed(1)}) than absorbed (${fighter.SApM?.toFixed(1)}).` : 
                    `Takes more strikes (${fighter.SApM?.toFixed(1)}) than lands (${fighter.SLpM?.toFixed(1)}).`
                  }
                  {fighter.win_by_KO_TKO > fighter.win_by_SUB && fighter.win_by_KO_TKO > fighter.win_by_DEC ? 
                    ` Primarily wins by KO/TKO (${fighter.win_by_KO_TKO}).` : 
                    fighter.win_by_SUB > fighter.win_by_KO_TKO && fighter.win_by_SUB > fighter.win_by_DEC ?
                    ` Primarily wins by submission (${fighter.win_by_SUB}).` :
                    ` Primarily wins by decision (${fighter.win_by_DEC}).`
                  }
                </p>
              </div>
            </div>
          </div>
          
          {/* Create Prediction Button */}
          <motion.div variants={item} className="mt-8">
            <div className="card backdrop-blur-sm bg-gray-900/60 border border-gray-800 shadow-xl rounded-lg p-6">
              <h2 className="text-xl font-bold mb-4 text-gray-200">Fight Predictions</h2>
              <p className="text-gray-300 mb-6">
                Want to see how {fighter.name} would perform against another fighter? Use our advanced prediction engine to simulate a matchup.
              </p>
              <Link 
                to="/predict" 
                className="block text-center py-3 px-6 bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 text-white font-bold rounded-lg shadow-lg transition-all duration-300 transform hover:scale-105"
              >
                Create a Fight Prediction
              </Link>
            </div>
          </motion.div>
        </motion.div>
      </motion.div>
    </motion.div>
  );
};

export default FighterDetail;
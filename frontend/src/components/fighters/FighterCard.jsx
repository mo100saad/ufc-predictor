import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import FighterImage from './FighterImage';

const FighterCard = ({ fighter }) => {
  if (!fighter) return null;
  
  // Calculate win percentage
  const totalFights = (fighter.wins || 0) + (fighter.losses || 0);
  const winPercentage = totalFights > 0 ? ((fighter.wins || 0) / totalFights * 100).toFixed(0) : 0;
  
  // Determine fighter style based on stats with more accurate detection and fallback
  const getFighterStyle = () => {
    const strikingRatio = (fighter.SLpM || 0) / 5; // Normalized to 0-1 where 5 is high
    const grapplingRatio = ((fighter.td_avg || 0) + (fighter.sub_avg || 0)) / 6; // Normalized
    
    // First check if fighter has a style property directly
    if (fighter.style) return fighter.style;
    
    // Otherwise determine based on strike/grappling metrics
    if (strikingRatio > 0.7 && grapplingRatio < 0.3) return "Striker";
    if (strikingRatio < 0.3 && grapplingRatio > 0.7) return "Grappler";
    if (strikingRatio > 0.5 && grapplingRatio > 0.5) return "Well-Rounded";
    if (strikingRatio > 0.4 && grapplingRatio > 0.3) return "Balanced";
    
    // Use the style from stats_view in Fighters.jsx as a final fallback
    if (strikingRatio > 0.3 || grapplingRatio > 0.2) return "Balanced";
    return "Striker"; // Default to striker instead of "Unknown Style"
  };
  
  // Determine fighter's weight class based on kilograms with slight adjustments for borderline decimals
  const getWeightClass = (weightInKg) => {
    if (!weightInKg) return "Unknown";

    // Adjusted thresholds (all slightly increased)
    if (weightInKg <= 56.8)  return "Flyweight";         // originally 56.7 kg (~125 lbs)
    if (weightInKg <= 61.3)  return "Bantamweight";      // originally 61.2 kg (~135 lbs)
    if (weightInKg <= 65.9)  return "Featherweight";     // originally 65.8 kg (~145 lbs)
    if (weightInKg <= 70.32) return "Lightweight";       // originally 70.31 kg (~155 lbs)
    if (weightInKg <= 77.2)  return "Welterweight";      // originally 77.1 kg (~170 lbs)
    if (weightInKg <= 84.0)  return "Middleweight";      // originally 83.9 kg (~185 lbs)
    if (weightInKg <= 93.1)  return "Light Heavyweight"; // originally 93.0 kg (~205 lbs)
    return "Heavyweight";                                // >93.1 kg
  };

  // Background style for cards - varies by weight class for visual distinction
  const getBgGradient = (weightClass) => {
    const gradients = {
      "Flyweight": "from-purple-900/30 to-indigo-900/20",
      "Bantamweight": "from-indigo-900/30 to-blue-900/20",
      "Featherweight": "from-blue-900/30 to-teal-900/20",
      "Lightweight": "from-teal-900/30 to-green-900/20",
      "Welterweight": "from-green-900/30 to-yellow-900/20",
      "Middleweight": "from-yellow-900/30 to-orange-900/20",
      "Light Heavyweight": "from-orange-900/30 to-red-900/20",
      "Heavyweight": "from-red-900/30 to-rose-900/20",
      "Unknown": "from-gray-900/30 to-gray-800/20"
    };
    
    return gradients[weightClass] || gradients["Unknown"];
  };

  return (
    <motion.div 
      className="group h-full"
      whileHover={{ y: -5 }}
      transition={{ type: "spring", stiffness: 300 }}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
    >
      <Link 
        to={`/fighters/${encodeURIComponent(fighter.name)}`}
        className="block h-full"
      >
        <div className={`relative h-full backdrop-blur-sm bg-gradient-to-br ${getBgGradient(getWeightClass(fighter.weight))} border border-gray-800 shadow-lg hover:shadow-xl rounded-lg overflow-hidden transition-all duration-300 hover:border-gray-700 group-hover:bg-opacity-90`}>
          
          {/* Corner accent */}
          <div className="absolute top-0 left-0 w-16 h-16">
            <div className="absolute transform rotate-45 bg-gradient-to-r from-red-500/40 to-blue-500/40 w-16 h-16 -top-8 -left-8"></div>
          </div>
          
          {/* Card Content */}
          <div className="p-5">
            <div className="flex items-start justify-between">
              {/* Left column - Fighter details */}
              <div className="flex-1">
                {/* Fighter Name - With truncation for long names */}
                <h3 className="text-xl font-bold text-white mb-1 group-hover:text-blue-400 transition-colors duration-300 truncate max-w-[160px]" title={fighter.name}>
                  {fighter.name}
                </h3>
                
                {/* Record and Division */}
                <div className="flex flex-wrap items-center gap-2 mb-3">
                  <span className="text-sm font-medium bg-gray-800/70 text-gray-300 px-2 py-1 rounded-md">
                    {fighter.wins || 0}-{fighter.losses || 0}{fighter.draws ? `-${fighter.draws}` : ''}
                  </span>
                  {winPercentage > 0 && (
                    <span className={`text-xs px-1.5 py-0.5 rounded-md ${
                      winPercentage >= 70 ? 'bg-green-900/60 text-green-400' : 
                      winPercentage >= 50 ? 'bg-blue-900/60 text-blue-400' : 
                      'bg-red-900/60 text-red-400'
                    }`}>
                      {winPercentage}% Wins
                    </span>
                  )}
                  <span className="text-xs px-1.5 py-0.5 rounded-md bg-gray-800/70 text-gray-400">
                    {getWeightClass(fighter.weight)}
                  </span>
                </div>
              </div>
            </div>
            
            {/* Improved fighter image display area */}
            <div className="w-full h-60 my-4 flex justify-center items-start bg-gradient-to-b from-gray-800/10 to-gray-900/40 rounded-lg">
              <FighterImage 
                src={fighter.image_url} 
                alt={fighter.name}
                size="md" 
                rounded={false}
                className="object-contain object-top h-full"
                withBorder={false}
              />
            </div>
            
            {/* Fighter Stats - horizontal layout for better space usage */}
            <div className="flex gap-2 mb-4">
              <div className="flex-1 text-center p-2 bg-gray-800/70 backdrop-blur-sm rounded-lg transition-all duration-300 hover:bg-gray-800/90">
                <div className="text-lg font-bold text-red-500">{fighter.SLpM?.toFixed(1) || 'N/A'}</div>
                <div className="text-xs text-gray-400">Strikes/Min</div>
              </div>
              
              <div className="flex-1 text-center p-2 bg-gray-800/70 backdrop-blur-sm rounded-lg transition-all duration-300 hover:bg-gray-800/90">
                <div className="text-lg font-bold text-blue-500">{fighter.td_avg?.toFixed(1) || 'N/A'}</div>
                <div className="text-xs text-gray-400">Takedowns/Fight</div>
              </div>
            </div>
            
            {/* Fighter Style & View Details */}
            <div className="flex justify-between items-center">
              <span className="text-xs py-1 px-2 rounded-full bg-gray-800/50 text-gray-400">
                <span className="text-gray-300">{getFighterStyle()}</span>
              </span>
              
              <motion.span 
                className="text-blue-400 text-xs font-medium"
                initial={{ opacity: 0, x: 10 }}
                whileHover={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.2 }}
              >
                View Profile →
              </motion.span>
            </div>
          </div>
        </div>
      </Link>
    </motion.div>
  );
};

export default FighterCard;
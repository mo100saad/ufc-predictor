import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { fighterService } from '../services/api';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorAlert from '../components/common/ErrorAlert';

const FighterDetail = () => {
  const { name } = useParams();
  const [fighter, setFighter] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadFighter = async () => {
      try {
        setLoading(true);
        const data = await fighterService.getFighterByName(name);
        setFighter(data);
        setLoading(false);
      } catch (err) {
        console.error("Failed to load fighter:", err);
        setError(`Failed to load data for ${name}. Please try again later.`);
        setLoading(false);
      }
    };

    loadFighter();
  }, [name]);

  if (loading) {
    return <LoadingSpinner text={`Loading data for ${name}...`} />;
  }

  if (error) {
    return (
      <div>
        <ErrorAlert message={error} />
        <Link to="/fighters" className="text-ufc-blue hover:text-ufc-red mt-2 inline-block">
          ← Back to fighters
        </Link>
      </div>
    );
  }

  if (!fighter) {
    return (
      <div className="text-center py-10">
        <p className="text-gray-400">Fighter not found</p>
        <Link to="/fighters" className="text-ufc-blue hover:text-ufc-red mt-2 inline-block">
          ← Back to fighters
        </Link>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-6">
        <Link to="/fighters" className="text-ufc-blue hover:text-ufc-red">
          ← Back to fighters
        </Link>
      </div>
      
      <div className="card mb-6">
        <h1 className="text-3xl font-bold mb-2">{fighter.name}</h1>
        <div className="text-lg text-gray-300 mb-4">
          Record: <span className="font-medium">{fighter.wins || 0}-{fighter.losses || 0}-{fighter.draws || 0}</span>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h2 className="text-xl font-bold mb-3">Physical Attributes</h2>
            <table className="w-full">
              <tbody>
                <tr className="border-b border-gray-700">
                  <td className="py-2 text-gray-400">Height</td>
                  <td className="py-2">{fighter.height ? `${fighter.height} cm` : 'N/A'}</td>
                </tr>
                <tr className="border-b border-gray-700">
                  <td className="py-2 text-gray-400">Weight</td>
                  <td className="py-2">{fighter.weight ? `${fighter.weight} lbs` : 'N/A'}</td>
                </tr>
                <tr className="border-b border-gray-700">
                  <td className="py-2 text-gray-400">Reach</td>
                  <td className="py-2">{fighter.reach ? `${fighter.reach} cm` : 'N/A'}</td>
                </tr>
                <tr>
                  <td className="py-2 text-gray-400">Stance</td>
                  <td className="py-2">{fighter.stance || 'N/A'}</td>
                </tr>
              </tbody>
            </table>
          </div>
          
          <div>
            <h2 className="text-xl font-bold mb-3">Fighting Statistics</h2>
            <table className="w-full">
              <tbody>
                <tr className="border-b border-gray-700">
                  <td className="py-2 text-gray-400">Striking Accuracy</td>
                  <td className="py-2">{fighter.sig_strikes_per_min ? `${fighter.sig_strikes_per_min.toFixed(1)}/min` : 'N/A'}</td>
                </tr>
                <tr className="border-b border-gray-700">
                  <td className="py-2 text-gray-400">Takedown Average</td>
                  <td className="py-2">{fighter.takedown_avg ? `${fighter.takedown_avg.toFixed(1)}/fight` : 'N/A'}</td>
                </tr>
                <tr className="border-b border-gray-700">
                  <td className="py-2 text-gray-400">Submission Average</td>
                  <td className="py-2">{fighter.sub_avg ? `${fighter.sub_avg.toFixed(1)}/fight` : 'N/A'}</td>
                </tr>
                <tr>
                  <td className="py-2 text-gray-400">Win Streak</td>
                  <td className="py-2">{fighter.win_streak || '0'}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card">
          <h2 className="text-xl font-bold mb-3">Win Breakdown</h2>
          <div className="flex justify-around text-center">
            <div>
              <div className="text-2xl font-bold text-ufc-red">{fighter.win_by_KO_TKO || '0'}</div>
              <div className="text-sm text-gray-400">KO/TKO</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-ufc-blue">{fighter.win_by_SUB || '0'}</div>
              <div className="text-sm text-gray-400">Submission</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-ufc-gold">{fighter.win_by_DEC || '0'}</div>
              <div className="text-sm text-gray-400">Decision</div>
            </div>
          </div>
        </div>
        
        <div className="card">
          <h2 className="text-xl font-bold mb-3">Prediction</h2>
          <p className="text-gray-300 mb-4">
            Want to see how {fighter.name} would perform against another fighter?
          </p>
          <Link 
            to="/predict" 
            className="block text-center btn-primary"
          >
            Create a Fight Prediction
          </Link>
        </div>
      </div>
    </div>
  );
};

export default FighterDetail;
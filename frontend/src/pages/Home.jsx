import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { fightService } from '../services/api';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorAlert from '../components/common/ErrorAlert';

const Home = () => {
  const [recentFights, setRecentFights] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadRecentFights = async () => {
      try {
        setLoading(true);
        const fights = await fightService.getAllFights();
        setRecentFights(fights.slice(0, 5)); // Get 5 most recent fights
        setLoading(false);
      } catch (err) {
        console.error("Failed to load recent fights:", err);
        setError("Failed to load recent fights. Please try again later.");
        setLoading(false);
      }
    };

    loadRecentFights();
  }, []);

  return (
    <div>
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold mb-4">UFC Fight Predictor</h1>
        <p className="text-lg text-gray-300 max-w-2xl mx-auto">
          Predict UFC fight outcomes with our advanced machine learning model. 
          Analyze fighter statistics and get accurate predictions for upcoming matches.
        </p>
        <div className="mt-8 flex flex-wrap justify-center gap-4">
          <Link to="/predict" className="btn-primary">
            Predict a Fight
          </Link>
          <Link to="/fighters" className="btn-secondary">
            Browse Fighters
          </Link>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-12">
        <div className="card">
          <h2 className="text-2xl font-bold mb-4">How It Works</h2>
          <p className="text-gray-300 mb-4">
            Our prediction model is trained on thousands of UFC fights and considers various factors:
          </p>
          <ul className="list-disc pl-5 text-gray-300 space-y-2">
            <li>Fighter statistics (win/loss records, striking accuracy)</li>
            <li>Physical attributes (reach, height, weight)</li>
            <li>Fighting style matchups</li>
            <li>Recent performance trends</li>
          </ul>
        </div>

        <div className="card">
          <h2 className="text-2xl font-bold mb-4">Recent Fights</h2>
          {loading ? (
            <LoadingSpinner text="Loading recent fights..." />
          ) : error ? (
            <ErrorAlert message={error} />
          ) : (
            <ul className="divide-y divide-gray-700">
              {recentFights.map((fight) => (
                <li key={fight.id} className="py-3">
                  <div className="flex justify-between">
                    <div className="flex items-center">
                      <span className={`font-medium ${fight.winner_name === fight.fighter1_name ? 'text-ufc-red' : ''}`}>
                        {fight.fighter1_name}
                      </span>
                      <span className="mx-2 text-gray-400">vs</span>
                      <span className={`font-medium ${fight.winner_name === fight.fighter2_name ? 'text-ufc-blue' : ''}`}>
                        {fight.fighter2_name}
                      </span>
                    </div>
                    <span className="text-sm text-gray-400">{fight.date}</span>
                  </div>
                  <div className="text-sm text-gray-400 mt-1">
                    Winner: {fight.winner_name} • Method: {fight.method}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
};

export default Home;
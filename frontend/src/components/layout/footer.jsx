import React from 'react';
import { motion } from 'framer-motion';

const Footer = () => {
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className="relative mt-20">
      {/* Gradient divider */}
      <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-red-500 via-purple-500 to-blue-500"></div>
      
      {/* Background effect */}
      <div className="absolute inset-0 bg-gradient-to-t from-black to-gray-900 -z-10"></div>
      
      <div className="container mx-auto px-4 py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Logo and copyright section */}
          <div className="flex flex-col items-center md:items-start">
            <div className="flex items-center mb-4">
              <span className="text-2xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-red-500 to-red-700">UFC</span>
              <span className="text-2xl font-extrabold ml-2 text-white">Predictor</span>
            </div>
            <p className="text-gray-400 text-sm text-center md:text-left">
              &copy; {currentYear} UFC Fight Predictor<br />
              Powered by Mo Saad's Machine Learning
            </p>
          </div>
          
          {/* Quick links section */}
          <div className="text-center">
            <h3 className="text-lg font-bold text-white mb-4">Quick Links</h3>
            <ul className="space-y-2">
              {[
                { label: 'Home', path: '/' },
                { label: 'Fighters Database', path: '/fighters' },
                { label: 'Fight Predictions', path: '/predict' }
              ].map((link, index) => (
                <li key={index}>
                  <a 
                    href={link.path} 
                    className="text-gray-400 hover:text-white transition-colors duration-300"
                  >
                    {link.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>
          
          {/* Connect section */}
          <div className="text-center md:text-right">
            <h3 className="text-lg font-bold text-white mb-4">Connect</h3>
            
            <motion.a 
              href="https://github.com/mo100saad/ufc-predictor"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg transition-colors duration-300"
              whileHover={{ scale: 1.05 }}
              transition={{ duration: 0.2 }}
            >
              <svg className="h-5 w-5 mr-2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
              </svg>
              GitHub
            </motion.a>
            
            <p className="mt-6 text-gray-500 text-sm">
              This application is for educational purposes only don't gamble.
            </p>
          </div>
        </div>
        
        <div className="mt-10 pt-6 border-t border-gray-800 text-center">
          <p className="text-gray-500 text-xs">
            UFC is a registered trademark of Zuffa, LLC. This site is not affiliated with UFC or Zuffa, LLC but is taken as inspiration by Mohammad Saad.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
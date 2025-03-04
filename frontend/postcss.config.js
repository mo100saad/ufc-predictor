/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'ufc-red': '#D20A0A',
        'ufc-blue': '#1277BC',
        'ufc-gold': '#FFD700',
        'dark-bg': '#121212',
        'card-bg': '#1E1E1E',
      },
    },
  },
  plugins: [],
}
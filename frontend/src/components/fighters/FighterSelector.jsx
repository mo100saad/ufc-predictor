import React from 'react';
import { Field } from 'formik';

const FighterSelector = ({ name, fighters, cornerClass }) => {
  return (
    <Field
      as="select"
      name={name}
      className={`w-full p-2 bg-gray-800 border border-gray-700 rounded ${cornerClass}`}
    >
      <option value="">Select a fighter</option>
      {fighters.map(fighter => (
        <option key={fighter.id || fighter.name} value={fighter.name}>
          {fighter.name}
        </option>
      ))}
    </Field>
  );
};

export default FighterSelector;
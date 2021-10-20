import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom/extend-expect';
import SigmaGraph from './SigmaGraph';

describe('<SigmaGraph />', () => {
  test('it should mount', () => {
    render(<SigmaGraph />);
    
    const sigmaGraph = screen.getByTestId('SigmaGraph');

    expect(sigmaGraph).toBeInTheDocument();
  });
});
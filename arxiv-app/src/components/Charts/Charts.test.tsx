import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom/extend-expect';
import Charts from './Charts';

describe('<Charts />', () => {
  test('it should mount', () => {
    render(<Charts />);
    
    const charts = screen.getByTestId('Charts');

    expect(charts).toBeInTheDocument();
  });
});
import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom/extend-expect';
import Statistics from './Statistics';

describe('<Statistics />', () => {
  test('it should mount', () => {
    render(<Statistics />);
    
    const statistics = screen.getByTestId('Statistics');

    expect(statistics).toBeInTheDocument();
  });
});
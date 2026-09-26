import React from 'react';
import { render, screen } from '@testing-library/react';
import { expect, test } from 'vitest';
import App from './App';

test('renders title', () => {
  render(<App />);
  expect(screen.getByText(/Seen'emAll/i)).toBeInTheDocument();
});

import React, { useState } from 'react';
import { Button, Card, Container, Form, Row, Col, Spinner } from 'react-bootstrap';

interface Genre {
  id: number;
  name: string;
}

interface WatchOption {
  service: string;
  url: string;
}

interface Recommendation {
  id: number;
  tmdb_id: number;
  media_type: 'movie' | 'tv';
  title: string;
  overview: string;
  poster_url: string | null;
  runtime: number | null;
  original_language: string;
  genres: Genre[];
  release_year: number;
  watch_url?: string;
  watch_options?: WatchOption[];
  score: number;
  explanation: string;
}

function App() {
  const [userId, setUserId] = useState('u1'); // Default user ID
  const [query, setQuery] = useState('');
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [diversify, setDiversify] = useState(true);

  const initializeUserHistory = async () => {
    try {
      await fetch(`${process.env.REACT_APP_API_URL}/user/history`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          items: []  // Start with empty history
        }),
      });
    } catch (error) {
      console.error('Error initializing user history:', error);
    }
  };

  const getRecommendations = async () => {
    setLoading(true);
    setError(null);
    try {
      // Try to initialize user history first (if not already done)
      console.log('Initializing user history for:', userId);
      await initializeUserHistory();

      const params = new URLSearchParams();
      params.append('user_id', userId);
      if (query.trim()) {
        params.append('query', query);
      }
      if (!diversify) {
        params.append('diversify', 'false');
      }

      const url = `${process.env.REACT_APP_API_URL}/recommend?${params.toString()}`;
      console.log('Fetching recommendations from:', url);

      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      if (!response.ok) {
        const errorData = await response.json();
        console.error('API Error:', errorData);
        throw new Error(`Failed to fetch recommendations: ${JSON.stringify(errorData)}`);
      }
      const data = await response.json();
      console.log('API Response:', data);
      // The API might return the recommendations directly, not in an 'items' property
      setRecommendations(Array.isArray(data) ? data : data.items || []);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
      setError('Failed to load recommendations. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleWatch = async (recommendationId: number) => {
    try {
      await fetch(`${process.env.REACT_APP_API_URL}/user/history`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          items: [recommendationId],
          event_type: 'watched',
          weight: 1
        }),
      });
      // Refresh recommendations after marking as watched
      getRecommendations();
    } catch (error) {
      console.error('Error marking as watched:', error);
    }
  };

  return (
    <Container className="py-4">
      <h1 className="text-center mb-4">Seen'emAll</h1>

      <Form className="mb-4" onSubmit={(e) => {
        e.preventDefault();
        getRecommendations();
      }}>
        <Row className="justify-content-center mb-3">
          <Col md={2}>
            <Form.Group>
              <Form.Label>User ID</Form.Label>
              <Form.Control
                type="text"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                placeholder="e.g., u1"
              />
            </Form.Group>
          </Col>
        </Row>
        <Row className="justify-content-center">
          <Col md={8}>
            <Form.Group className="mb-3">
              <Form.Control
                type="text"
                placeholder="What kind of movie or show are you looking for?"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
              <Form.Text className="text-muted">
                Try: "sci-fi movies like Inception" or "comedy shows like The Office"
              </Form.Text>
            </Form.Group>
            <Form.Check
              type="switch"
              id="diversify-toggle"
              label="Enable diversity boosts"
              checked={diversify}
              onChange={(e) => setDiversify(e.target.checked)}
            />
          </Col>
          <Col md={2}>
            <Button
              variant="primary"
              type="submit"
              className="w-100"
              disabled={loading}
            >
              {loading ? (
                <>
                  <Spinner
                    as="span"
                    animation="border"
                    size="sm"
                    role="status"
                    aria-hidden="true"
                    className="me-2"
                  />
                  Loading...
                </>
              ) : (
                'Get Recommendations'
              )}
            </Button>
          </Col>
        </Row>
      </Form>

      {error && (
        <div className="alert alert-danger text-center" role="alert">
          {error}
        </div>
      )}

      <Row xs={1} md={2} lg={3} className="g-4">
        {recommendations.map((rec) => (
          <Col key={rec.id}>
            <Card className="h-100">
              {rec.poster_url && (
                <Card.Img
                  variant="top"
                  src={rec.poster_url}
                  alt={`${rec.title} poster`}
                  style={{ height: '300px', objectFit: 'cover' }}
                />
              )}
              <Card.Body className="d-flex flex-column">
                <Card.Title>{rec.title}</Card.Title>
                <Card.Subtitle className="mb-2 text-muted">
                  {rec.media_type.toUpperCase()} • {rec.release_year} • {rec.runtime ? `${rec.runtime}min` : 'N/A'}
                </Card.Subtitle>
                <Card.Text>{rec.overview}</Card.Text>
                <small className="text-muted mb-2 d-block">
                  {rec.genres.map(genre => genre.name).join(', ')}
                </small>
                <div className="mt-auto">
                  {rec.watch_options && rec.watch_options.length > 0 ? (
                    <div className="mb-2">
                      <div className="dropdown">
                        <Button
                          variant="success"
                          className="dropdown-toggle"
                          data-bs-toggle="dropdown"
                          aria-expanded="false"
                        >
                          Watch Now
                        </Button>
                        <ul className="dropdown-menu">
                          {rec.watch_options.map((opt, idx) => (
                            <li key={idx}>
                              <a
                                className="dropdown-item"
                                href={opt.url}
                                target="_blank"
                                rel="noopener noreferrer"
                              >
                                {opt.service.toUpperCase()}
                              </a>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  ) : (
                    <Button
                      variant="secondary"
                      className="me-2 mb-2"
                      disabled
                    >
                      No Streaming Link
                    </Button>
                  )}
                  <Button
                    variant="outline-primary"
                    onClick={() => handleWatch(rec.id)}
                  >
                    Mark as Watched
                  </Button>
                </div>
              </Card.Body>
            </Card>
          </Col>
        ))}
      </Row>
    </Container>
  );
}

export default App;

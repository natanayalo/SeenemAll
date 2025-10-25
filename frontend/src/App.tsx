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
  vote_average?: number | null;
  vote_count?: number | null;
  popularity?: number | null;
}

function App() {
  const defaultMixerAnn = 0.5;
  const defaultMixerCollab = 0.3;
  const defaultMixerTrending = 0.4;
  const defaultMixerPopularity = 0.25;
  const defaultMixerNovelty = 0.1;
  const defaultMixerVote = 0.2;
  const defaultAnnDescriptionWeight = 1.2;
  const defaultRewriteTextWeight = 1.0;
  const [userId, setUserId] = useState('u1'); // Default user ID
  const [query, setQuery] = useState('');
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [diversify, setDiversify] = useState(true);
  const [useLlmIntent, setUseLlmIntent] = useState(true);
  const [manualAnnDescription, setManualAnnDescription] = useState('');
  const [manualRewrite, setManualRewrite] = useState('');
  const [annWeight, setAnnWeight] = useState(defaultAnnDescriptionWeight);
  const [rewriteWeight, setRewriteWeight] = useState(defaultRewriteTextWeight);
  const [genreOverride, setGenreOverride] = useState('');
  const [useMixerOverrides, setUseMixerOverrides] = useState(false);
  const [mixerAnnWeight, setMixerAnnWeight] = useState(defaultMixerAnn);
  const [mixerCollabWeight, setMixerCollabWeight] = useState(defaultMixerCollab);
  const [mixerTrendingWeight, setMixerTrendingWeight] = useState(defaultMixerTrending);
  const [mixerPopularityWeight, setMixerPopularityWeight] = useState(defaultMixerPopularity);
  const [mixerNoveltyWeight, setMixerNoveltyWeight] = useState(defaultMixerNovelty);
  const [mixerVoteWeight, setMixerVoteWeight] = useState(defaultMixerVote);

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
      if (!useLlmIntent) {
        params.append('use_llm_intent', 'false');
      }
      if (manualAnnDescription.trim()) {
        params.append('ann_description_override', manualAnnDescription.trim());
      }
      if (manualRewrite.trim()) {
        params.append('rewrite_override', manualRewrite.trim());
      }
      if (genreOverride.trim()) {
        params.append('genre_override', genreOverride.trim());
      }
      const hasCustomAnnWeight =
        Math.abs(annWeight - defaultAnnDescriptionWeight) > 0.001;
      if (hasCustomAnnWeight) {
        params.append('ann_weight_override', annWeight.toString());
      }
      const hasCustomRewriteWeight =
        Math.abs(rewriteWeight - defaultRewriteTextWeight) > 0.001;
      if (hasCustomRewriteWeight) {
        params.append('rewrite_weight_override', rewriteWeight.toString());
      }
      if (useMixerOverrides) {
        params.append('mixer_ann_weight', mixerAnnWeight.toString());
        params.append('mixer_collab_weight', mixerCollabWeight.toString());
        params.append('mixer_trending_weight', mixerTrendingWeight.toString());
        params.append('mixer_popularity_weight', mixerPopularityWeight.toString());
        params.append('mixer_novelty_weight', mixerNoveltyWeight.toString());
        params.append('mixer_vote_weight', mixerVoteWeight.toString());
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
            <Form.Check
              type="switch"
              id="use-llm-toggle"
              label="Use LLM intent parser"
              checked={useLlmIntent}
              onChange={(e) => setUseLlmIntent(e.target.checked)}
              className="mt-2"
            />
            <Form.Text className="text-muted">
              Neighbor weight is controlled by MIXER_COLLAB_WEIGHT (default 0.3).
            </Form.Text>
            <Form.Check
              type="switch"
              id="mixer-override-toggle"
              label="Customize mixer weights"
              checked={useMixerOverrides}
              onChange={(e) => setUseMixerOverrides(e.target.checked)}
              className="mt-2"
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
        <Row className="justify-content-center mt-3">
          <Col md={4}>
            <Form.Group>
              <Form.Label>Manual ANN Description</Form.Label>
              <Form.Control
                as="textarea"
                rows={2}
                value={manualAnnDescription}
                onChange={(e) => setManualAnnDescription(e.target.value)}
                placeholder="Optional override e.g. 'A deadly survival game for a cash prize'"
              />
            </Form.Group>
          </Col>
          <Col md={4}>
            <Form.Group>
              <Form.Label>Manual Rewrite Text</Form.Label>
              <Form.Control
                type="text"
                value={manualRewrite}
                onChange={(e) => setManualRewrite(e.target.value)}
                placeholder="Optional override e.g. 'sci-fi survival games'"
              />
            </Form.Group>
          </Col>
          <Col md={4}>
            <Form.Group>
              <Form.Label>Genre Override</Form.Label>
              <Form.Control
                type="text"
                value={genreOverride}
                onChange={(e) => setGenreOverride(e.target.value)}
                placeholder="e.g., Drama, Sci-Fi"
              />
              <Form.Text className="text-muted">
                When set, replaces inferred genres with this comma-separated list.
              </Form.Text>
            </Form.Group>
          </Col>
        </Row>
        <Row className="justify-content-center mt-2">
          <Col md={3}>
            <Form.Group>
              <Form.Label>ANN Description Weight ({annWeight.toFixed(2)})</Form.Label>
              <Form.Range
                min={0}
                max={2}
                step={0.1}
                value={annWeight}
                onChange={(e) => setAnnWeight(parseFloat(e.target.value))}
              />
            </Form.Group>
          </Col>
          <Col md={3}>
            <Form.Group>
              <Form.Label>Rewrite Text Weight ({rewriteWeight.toFixed(2)})</Form.Label>
              <Form.Range
                min={0}
                max={2}
                step={0.1}
                value={rewriteWeight}
                onChange={(e) => setRewriteWeight(parseFloat(e.target.value))}
              />
            </Form.Group>
          </Col>
        </Row>
      </Form>

      {useMixerOverrides && (
        <Row className="justify-content-center mt-3">
          <Col md={10}>
            <Form.Label>Hybrid / Mixer Weights</Form.Label>
            <Form.Text className="text-muted d-block mb-2">
              Adjust how ANN, collaborative, trending, popularity, and novelty signals influence ranking.
            </Form.Text>
            <Row className="gy-3">
              <Col md={2} sm={4} xs={6}>
                <Form.Group>
                  <Form.Label>ANN ({mixerAnnWeight.toFixed(2)})</Form.Label>
                  <Form.Range
                    min={0}
                    max={2}
                    step={0.05}
                    value={mixerAnnWeight}
                    onChange={(e) => setMixerAnnWeight(parseFloat(e.target.value))}
                  />
                </Form.Group>
              </Col>
              <Col md={2} sm={4} xs={6}>
                <Form.Group>
                  <Form.Label>Collab ({mixerCollabWeight.toFixed(2)})</Form.Label>
                  <Form.Range
                    min={0}
                    max={2}
                    step={0.05}
                    value={mixerCollabWeight}
                    onChange={(e) => setMixerCollabWeight(parseFloat(e.target.value))}
                  />
                </Form.Group>
              </Col>
              <Col md={2} sm={4} xs={6}>
                <Form.Group>
                  <Form.Label>Trending ({mixerTrendingWeight.toFixed(2)})</Form.Label>
                  <Form.Range
                    min={0}
                    max={2}
                    step={0.05}
                    value={mixerTrendingWeight}
                    onChange={(e) => setMixerTrendingWeight(parseFloat(e.target.value))}
                  />
                </Form.Group>
              </Col>
              <Col md={2} sm={4} xs={6}>
                <Form.Group>
                  <Form.Label>Popularity ({mixerPopularityWeight.toFixed(2)})</Form.Label>
                  <Form.Range
                    min={0}
                    max={2}
                    step={0.05}
                    value={mixerPopularityWeight}
                    onChange={(e) => setMixerPopularityWeight(parseFloat(e.target.value))}
                  />
                </Form.Group>
              </Col>
              <Col md={2} sm={4} xs={6}>
                <Form.Group>
                  <Form.Label>Novelty ({mixerNoveltyWeight.toFixed(2)})</Form.Label>
                  <Form.Range
                    min={0}
                    max={2}
                    step={0.05}
                    value={mixerNoveltyWeight}
                    onChange={(e) => setMixerNoveltyWeight(parseFloat(e.target.value))}
                  />
                </Form.Group>
              </Col>
              <Col md={2} sm={4} xs={6}>
                <Form.Group>
                  <Form.Label>Vote Count ({mixerVoteWeight.toFixed(2)})</Form.Label>
                  <Form.Range
                    min={0}
                    max={2}
                    step={0.05}
                    value={mixerVoteWeight}
                    onChange={(e) => setMixerVoteWeight(parseFloat(e.target.value))}
                  />
                </Form.Group>
              </Col>
            </Row>
          </Col>
        </Row>
      )}

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
                {(rec.vote_average || rec.vote_count || rec.popularity) && (
                  <div className="mb-2 small text-muted">
                    {rec.vote_average ? `⭐ ${rec.vote_average.toFixed(1)}/10` : '⭐ N/A'}
                    {typeof rec.vote_count === 'number' && rec.vote_count > 0 && (
                      <> · {rec.vote_count.toLocaleString()} votes</>
                    )}
                    {typeof rec.popularity === 'number' && (
                      <> · 🔥 Popularity {Math.round(rec.popularity)}</>
                    )}
                  </div>
                )}
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

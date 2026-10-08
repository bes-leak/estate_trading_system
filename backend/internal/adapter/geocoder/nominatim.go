package geocoder

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"time"

	"github.com/bes-leak/estate_trading_system/backend/internal/domain"
)

const nominatimURL = "https://nominatim.openstreetmap.org"
const agent = "estate-trading-system/0.1 (bees.23@uni-dubna.ru)"

type Nominatim struct {
	client    *http.Client
	baseURL   string
	userAgent string
}

func NewNominatim() *Nominatim {
	return &Nominatim{
		client:    &http.Client{Timeout: 5 * time.Second},
		baseURL:   nominatimURL,
		userAgent: agent,
	}
}

func (n *Nominatim) Geocode(ctx context.Context, address string) (domain.Coordinate, error) {
	params := url.Values{}
	params.Set("q", address)
	params.Set("format", "json")
	params.Set("limit", "1")

	fullURL := n.baseURL + "/search?" + params.Encode()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fullURL, nil)
	if err != nil {
		return domain.Coordinate{}, fmt.Errorf("geocoder: request: %w", err)
	}

	req.Header.Set("User-Agent", n.userAgent)

	resp, err := n.client.Do(req)

	if err != nil {
		return domain.Coordinate{}, fmt.Errorf("geocoder: network: %w", err)
	}

	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return domain.Coordinate{}, fmt.Errorf("geocoder: status: %d", resp.StatusCode)
	}

	var result []struct {
		Lat string `json:"lat"`
		Lon string `json:"lon"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return domain.Coordinate{}, fmt.Errorf("geocoder: decode response: %w", err)
	}

	if len(result) == 0 {
		return domain.Coordinate{}, domain.ErrAddressNotFound
	}

	lat, err := strconv.ParseFloat(result[0].Lat, 64)
	if err != nil {
		return domain.Coordinate{}, fmt.Errorf("geocoder: parse latitude %q: %w", result[0].Lat, err)
	}

	lon, err := strconv.ParseFloat(result[0].Lon, 64)
	if err != nil {
		return domain.Coordinate{}, fmt.Errorf("geocoder: parse longitude %q: %w", result[0].Lon, err)
	}

	return domain.Coordinate{Lat: lat, Lon: lon}, nil

}

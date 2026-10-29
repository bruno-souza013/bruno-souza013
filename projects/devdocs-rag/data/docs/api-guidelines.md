# REST API guidelines

Use nouns for resource paths and HTTP methods to communicate the operation. GET should read data, POST should create or trigger a processing operation, PUT should replace a resource, PATCH should update selected fields, and DELETE should remove a resource when supported.

Validate incoming request bodies and return clear error messages with appropriate HTTP status codes. A health endpoint should be lightweight and should not expose secrets or internal credentials. Document request and response schemas and include examples that can be copied into an API client.

For a long-running operation, consider timeouts and explicit failure states. Avoid logging passwords, access tokens, private documents, or other sensitive data. Tests should cover normal input, invalid input, empty results, and unavailable dependencies.

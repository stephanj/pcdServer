# Default Port 8090 Design

## Goal

Make PCD Server listen on TCP port 8090 when the user does not pass `--port`, and keep the executable help and current user documentation consistent with that default.

## Scope

- Change the application configuration default to 8090.
- Change the public `HttpServerOptions` default to 8090 so embedders get the same behavior.
- Update `pcd_server --help` to show `default 8090`.
- Update all current README commands, URLs, example startup output, and the CLI options table from 8080 to 8090.
- Add automated regression coverage for the help output and default declaration.

Historical implementation plans remain unchanged because they record the design at the time they were written.

## Design

The CLI and reusable HTTP server options will both use the literal default `8090`. The help output remains the canonical compact CLI reference and the README will reproduce the full output exactly, using `./build/pcd_server` as shown when running the local build.

An explicit `--port PORT` continues to override the default without any behavioral change. Port `0` remains supported through `HttpServerOptions` for tests that need an ephemeral listener.

## Testing

A command-level test will run `pcd_server --help` and assert that it exits successfully and reports `--port PORT           default 8090`. Existing tests and a full rebuild will verify that the server and embedded UI still compile. README references will be checked to ensure no current usage example retains port 8080.

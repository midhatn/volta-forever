#!/bin/sh
# volta-forever: install a release for the current user.
#
#   curl -fsSL https://raw.githubusercontent.com/nbald/volta-forever/main/install.sh | bash
#
# It downloads the archive of the version below from the project's releases, checks its
# SHA-256 against the release's SHA256SUMS, unpacks it under $VF_PREFIX, links the commands
# `vf` and `volta-forever` in $VF_BIN, runs `vf fetch` (which downloads NVIDIA's cuBLAS for
# CUDA 12.9.2 from NVIDIA's servers, 1 GB, under NVIDIA's licence) and ends with `vf doctor`.
# It installs no package of the system and asks for no password: what is missing is said.
#
# Settings, in the environment:
#   VF_VERSION        the version to install (default: the one this script was published with)
#   VF_PREFIX         where versions are unpacked (default: ~/.local/share/volta-forever;
#                     /opt/volta-forever for root)
#   VF_BIN            where the commands are linked (default: ~/.local/bin; /usr/local/bin
#                     for root)
#   VF_INSTALL_FETCH  0: do not run `vf fetch`
#   VF_INSTALL_BASE   where the archive and SHA256SUMS are taken from (default: the release
#                     of that version on GitHub): an https:// URL, or a file:// one
#   VF_INSTALL_ALLOW_HTTP
#                     1: take a VF_INSTALL_BASE of http:// too (a mirror on a network of
#                     your own)
#
# To remove it: delete $VF_PREFIX and the two links in $VF_BIN.
set -eu

say() { printf 'volta-forever: %s\n' "$*"; }
fail() { printf 'volta-forever: %s\n' "$*" >&2; exit 1; }

# (Everything is in a function that is called on the last line: a script cut short while it
# was being downloaded runs nothing.)
main() {
    version=${VF_VERSION:-0.1.6}
    case $version in *[!a-zA-Z0-9._+-]*) fail "invalid release version: $version" ;; esac
    name=volta-forever-$version-linux-x86_64
    archive=$name.tar.gz
    base=${VF_INSTALL_BASE:-https://github.com/nbald/volta-forever/releases/download/v$version}

    # --- what this needs of the machine ---
    [ "$(uname -s)" = Linux ] || fail "this is for Linux ($(uname -s) here)"
    [ "$(uname -m)" = x86_64 ] || fail "this is for x86-64 ($(uname -m) here)"
    libc=$(getconf GNU_LIBC_VERSION 2>/dev/null | sed -n 's/^glibc //p')
    [ -n "$libc" ] || fail "the C library of this system is not glibc: the binaries need glibc 2.34 or later"
    libc_major=${libc%%.*}
    libc_minor=${libc#*.}
    libc_minor=${libc_minor%%.*}
    if [ "$libc_major" -lt 2 ] || { [ "$libc_major" -eq 2 ] && [ "$libc_minor" -lt 34 ]; }; then
        fail "glibc $libc is too old: the binaries need 2.34 or later (Ubuntu 22.04, Debian 12, RHEL 9)"
    fi
    for tool in tar gzip sha256sum mktemp flock; do
        command -v "$tool" >/dev/null 2>&1 || fail "$tool is needed and was not found"
    done
    # (Over HTTPS alone, redirections included: SHA256SUMS comes from the same place as the
    # archive, and shows nothing if whoever is on the way can change both. A directory of this
    # machine, file://, has nobody on the way.)
    case $base in
        https://*) proto='=https' ;;
        file://*) proto='=file' ;;
        http://*)
            [ "${VF_INSTALL_ALLOW_HTTP:-0}" = 1 ] ||
                fail "VF_INSTALL_BASE is $base: over plain HTTP, whoever is on the way can change the archive and SHA256SUMS both (VF_INSTALL_ALLOW_HTTP=1 takes a mirror there all the same)"
            proto='=https,http' ;;
        *) fail "VF_INSTALL_BASE is $base: an https:// or a file:// URL is needed" ;;
    esac
    if command -v curl >/dev/null 2>&1; then
        download() { curl -fsSL --proto "$proto" --proto-redir "$proto" --retry 3 -o "$2" "$1"; }
    elif command -v wget >/dev/null 2>&1; then
        [ "$proto" != '=file' ] || fail "a file:// VF_INSTALL_BASE needs curl, which was not found"
        # (wget's --https-only holds only its recursive downloads to HTTPS: a redirection to
        # another scheme, which the headers that it prints show, is refused here.)
        download() {
            wget -q -S -O "$2" "$1" 2>"$2.headers" || return 1
            if [ "$proto" = '=https' ] && awk 'tolower($1) == "location:" && tolower($2) ~ /^[a-z][a-z0-9+.-]*:/ && tolower($2) !~ /^https:/ { found = 1 } END { exit !found }' "$2.headers"; then
                printf 'volta-forever: %s was redirected to another scheme than https://\n' "$1" >&2
                return 1
            fi
        }
    else
        fail "curl or wget is needed and neither was found"
    fi

    if [ "$(id -u)" = 0 ]; then
        prefix=${VF_PREFIX:-/opt/volta-forever}
        bin=${VF_BIN:-/usr/local/bin}
    else
        prefix=${VF_PREFIX:-${XDG_DATA_HOME:-$HOME/.local/share}/volta-forever}
        bin=${VF_BIN:-$HOME/.local/bin}
    fi

    # --- download and check ---
    work=$(mktemp -d "${TMPDIR:-/tmp}/volta-forever-install.XXXXXX") || fail "no temporary directory"
    stage=
    links=
    trap 'rm -rf -- "$work"
          [ -z "$stage" ] || rm -rf -- "$stage"
          [ -z "$links" ] || rm -rf -- "$links"' EXIT
    trap 'exit 1' HUP INT TERM
    say "downloading $archive from $base"
    download "$base/$archive" "$work/$archive" || fail "could not download $base/$archive"
    download "$base/SHA256SUMS" "$work/SHA256SUMS" || fail "could not download $base/SHA256SUMS"
    expected=$(awk -v file="$archive" '$2 == file || $2 == "*" file { print $1; exit }' "$work/SHA256SUMS")
    [ -n "$expected" ] || fail "SHA256SUMS of the release does not name $archive"
    got=$(sha256sum "$work/$archive" | cut -d' ' -f1)
    [ "$got" = "$expected" ] || fail "the SHA-256 of $archive is $got, and the release says $expected: nothing was installed"

    mkdir -p -- "$prefix" "$bin" || fail "cannot write in $prefix or $bin (VF_PREFIX and VF_BIN name other places)"
    prefix=$(CDPATH= cd -P -- "$prefix" && pwd)
    bin=$(CDPATH= cd -P -- "$bin" && pwd)
    stage=$(mktemp -d "$prefix/.vf-install.XXXXXXXX")
    links=$(mktemp -d "$bin/.vf-links.XXXXXXXX")
    tar -xzf "$work/$archive" -C "$stage" || fail "could not unpack $archive"
    pkg=$stage/$name
    [ -x "$pkg/bin/vf" ] || fail "$archive does not hold $name/bin/vf"

    # Extract independently. Serialize publication and share the CUDA payload across versions.
    exec 9>>"$prefix/.vf-install.lock"
    flock -x 9 || fail "cannot lock $prefix"
    mkdir -p -- "$prefix/.releases"
    # What killed installers left (the others remove their own): the directories not changed
    # for a day. A younger one may be another installer's, unpacking without the lock.
    find "$prefix/." "$bin/." ! -name . -prune -type d \( -name '.vf-install.????????' -o -name '.vf-links.????????' \) \
        -mtime +0 -exec rm -rf -- {} + || true
    release=$prefix/.releases/$name-$got
    if [ ! -e "$pkg/cuda-12.9.2" ] && [ ! -L "$pkg/cuda-12.9.2" ]; then
        cuda=$prefix/cuda-12.9.2
        if [ ! -d "$cuda" ]; then
            # A manually removed legacy cache may have left a dangling link.
            if [ -L "$cuda" ] && [ ! -e "$cuda" ]; then rm -- "$cuda"; fi
            legacy=
            for old in "$prefix/$name" "$prefix"/volta-forever-*-linux-x86_64; do
                [ -d "$old/cuda-12.9.2" ] || continue
                [ -n "$legacy" ] || legacy=$old/cuda-12.9.2
                if [ -f "$old/cuda-12.9.2/lib/libcublas.so.12" ]; then
                    legacy=$old/cuda-12.9.2
                    break
                fi
            done
            if [ -n "$legacy" ]; then
                # An older fetcher may still be writing here. Share the directory
                # itself so all versions use the same files and download locks.
                legacy=$(CDPATH= cd -P -- "$legacy" && pwd)
                ln -s -- "$legacy" "$stage/cuda"
            else
                mkdir -- "$stage/cuda"
            fi
            mv -T -- "$stage/cuda" "$cuda"
        fi
        # Older versions installed without a toolkit should also see later fetches.
        for old in "$prefix"/volta-forever-*-linux-x86_64; do
            if [ -d "$old" ] && [ ! -e "$old/cuda-12.9.2" ] && [ ! -L "$old/cuda-12.9.2" ]; then
                ln -sT -- "$cuda" "$old/cuda-12.9.2" || [ -d "$old/cuda-12.9.2" ]
            fi
        done
        ln -s -- ../../cuda-12.9.2 "$pkg/cuda-12.9.2"
    fi
    if [ ! -d "$release" ]; then
        mv -T -- "$pkg" "$release"
    fi
    # A legacy version directory can still be referenced by an existing venv.
    # New aliases and the active command switch by rename on their own filesystem.
    if [ ! -e "$prefix/$name" ] || [ -L "$prefix/$name" ]; then
        ln -s -- ".releases/$name-$got" "$stage/version"
        mv -Tf -- "$stage/version" "$prefix/$name"
    fi
    ln -s -- "$release/bin/vf" "$links/vf"
    ln -s -- vf "$links/volta-forever"
    if [ -e "$bin/vf" ]; then
        mv -Tf -- "$links/volta-forever" "$bin/volta-forever"
        mv -Tf -- "$links/vf" "$bin/vf"
    else
        mv -Tf -- "$links/vf" "$bin/vf"
        mv -Tf -- "$links/volta-forever" "$bin/volta-forever"
    fi
    flock -u 9
    exec 9<&-
    say "installed $(cat "$release/VERSION" 2>/dev/null || echo "$version") in $release"
    say "commands: $bin/vf, $bin/volta-forever"

    # --- NVIDIA's cuBLAS for CUDA 12.9.2 ---
    if [ "${VF_INSTALL_FETCH:-1}" != 0 ]; then
        if [ -f "$release/cuda-12.9.2/lib/libcublas.so.12" ]; then
            say "NVIDIA's cuBLAS 12.9.2 is there already"
        elif ! command -v python3 >/dev/null 2>&1; then
            say "python3 was not found: install it, then run: $bin/vf fetch"
        elif ! command -v xz >/dev/null 2>&1; then
            say "xz was not found (package xz-utils, or xz): install it, then run: $bin/vf fetch"
        else
            say "downloading NVIDIA's cuBLAS for CUDA 12.9.2 from NVIDIA's servers (1 GB), under NVIDIA's licence"
            if "$release/bin/vf" fetch; then
                say "NVIDIA's licence is in $release/cuda-12.9.2/LICENSE"
            else
                say "vf fetch did not end well: run it again with: $bin/vf fetch"
            fi
        fi
    fi

    # --- what is there ---
    echo
    "$release/bin/vf" doctor || true
    echo
    case ":$PATH:" in
        *":$bin:"*) ;;
        *) say "$bin is not on your PATH: add it with   export PATH=\"$bin:\$PATH\"" ;;
    esac
    say "run a program with   vf python3 your_program.py"
    say "a Python environment that runs through the layer by itself:   vf venv .venv"
}

main "$@"
